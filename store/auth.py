"""认证资料存储与解析。"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger("lingzhou.auth")


def _auth_base_dir() -> Path:
    """返回认证文件的运行时基目录，优先使用可写目录。"""
    env_override = os.getenv("LINGZHOU_DATA_DIR")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.extend(
        [
            Path("~/.lingzhou").expanduser(),
            Path.home() / ".cache" / "lingzhou",
            Path(tempfile.gettempdir()) / "lingzhou",
        ]
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except OSError as exc:
            _log.warning("[auth] 数据目录不可写，尝试下一个路径: %s (%s)", candidate, exc)
            continue
    fallback = candidates[-1]
    _log.warning("[auth] 采用兜底认证目录: %s", fallback)
    return fallback / "auth-fallback"


def _auth_path(relpath: str) -> Path:
    try:
        return (_auth_base_dir() / relpath).expanduser()
    except Exception:
        # 保险回退：若 data_dir 初始化失败，降级到 cache 下的 fallback 路径。
        return (Path.home() / ".cache" / "lingzhou" / "auth" / relpath).expanduser()


AUTH_PROFILES_PATH = _auth_path("auth-profiles.json")
COPILOT_TOKEN_CACHE_PATH = _auth_path("credentials/github-copilot.token.json")
GITHUB_DEVICE_AUTH_PATH = _auth_path("github-device.json")

COPILOT_PROFILE_ID = "copilot:default"
COPILOT_ENV_ORDER = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
BUILTIN_GITHUB_DEVICE_CLIENT_ID = ""


@dataclass(frozen=True)
class TokenResolution:
    token: str
    source: str
    profile_id: str | None = None


@dataclass(frozen=True)
class CopilotTokenCache:
    token: str
    expires_at_ms: int
    updated_at_ms: int


def mask_secret(secret: str) -> str:
    if len(secret) <= 12:
        return "*" * len(secret)
    return f"{secret[:8]}...{secret[-4:]}"


def load_auth_profiles(path: Path | None = None) -> dict[str, Any]:
    path = path or AUTH_PROFILES_PATH
    if not path.exists():
        return {"version": 1, "profiles": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "profiles": {}}
    if not isinstance(data, dict):
        return {"version": 1, "profiles": {}}
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    return {"version": int(data.get("version", 1)), "profiles": profiles}


def save_auth_profiles(data: dict[str, Any], path: Path | None = None) -> None:
    path = path or AUTH_PROFILES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


def get_auth_profile(profile_id: str, path: Path | None = None) -> dict[str, Any] | None:
    return load_auth_profiles(path).get("profiles", {}).get(profile_id)


def set_token_profile(
    *,
    profile_id: str = COPILOT_PROFILE_ID,
    provider: str,
    token: str,
    path: Path | None = None,
) -> None:
    data = load_auth_profiles(path)
    profiles = data.setdefault("profiles", {})
    profiles[profile_id] = {
        "type": "token",
        "provider": provider,
        "token": token,
    }
    save_auth_profiles(data, path)


def set_oauth_profile(
    *,
    profile_id: str,
    provider: str,
    tokens: dict[str, Any],
    auth_mode: str = "oauth",
    path: Path | None = None,
) -> None:
    data = load_auth_profiles(path)
    profiles = data.setdefault("profiles", {})
    profiles[profile_id] = {
        "type": "oauth",
        "provider": provider,
        "auth_mode": auth_mode,
        "tokens": tokens,
        "updated_at_ms": int(time.time() * 1000),
    }
    save_auth_profiles(data, path)


def load_github_device_client_id(path: Path | None = None) -> str:
    path = path or GITHUB_DEVICE_AUTH_PATH
    env_value = os.environ.get("LINGZHOU_GITHUB_CLIENT_ID", "").strip()
    if env_value:
        return env_value

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                client_id = str(data.get("client_id", "")).strip()
                if client_id:
                    return client_id
        except Exception:
            pass

    return BUILTIN_GITHUB_DEVICE_CLIENT_ID.strip()


def resolve_copilot_token(api_key_env: str = "GITHUB_TOKEN") -> TokenResolution | None:
    seen: set[str] = set()
    ordered_envs: list[str] = []
    for name in (*COPILOT_ENV_ORDER, api_key_env):
        if name and name not in seen:
            ordered_envs.append(name)
            seen.add(name)

    profile = get_auth_profile(COPILOT_PROFILE_ID)
    if profile and isinstance(profile, dict):
        token = str(profile.get("token", "")).strip()
        if token:
            return TokenResolution(token=token, source="auth-profile", profile_id=COPILOT_PROFILE_ID)

    for name in ordered_envs:
        token = os.environ.get(name, "").strip()
        if token:
            return TokenResolution(token=token, source=f"env:{name}")

    return None


def load_copilot_token_cache(path: Path | None = None) -> CopilotTokenCache | None:
    path = path or COPILOT_TOKEN_CACHE_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    token = str(data.get("token", "")).strip()
    expires = int(data.get("expiresAt", 0) or 0)
    updated = int(data.get("updatedAt", 0) or 0)
    if not token or expires <= 0:
        return None
    return CopilotTokenCache(token=token, expires_at_ms=expires, updated_at_ms=updated)


def save_copilot_token_cache(
    token: str,
    *,
    expires_at_ms: int,
    path: Path | None = None,
) -> None:
    path = path or COPILOT_TOKEN_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "token": token,
        "expiresAt": int(expires_at_ms),
        "updatedAt": int(time.time() * 1000),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


__all__ = [
    "AUTH_PROFILES_PATH",
    "BUILTIN_GITHUB_DEVICE_CLIENT_ID",
    "COPILOT_ENV_ORDER",
    "COPILOT_PROFILE_ID",
    "COPILOT_TOKEN_CACHE_PATH",
    "GITHUB_DEVICE_AUTH_PATH",
    "CopilotTokenCache",
    "TokenResolution",
    "get_auth_profile",
    "load_auth_profiles",
    "load_copilot_token_cache",
    "load_github_device_client_id",
    "mask_secret",
    "resolve_copilot_token",
    "save_auth_profiles",
    "save_copilot_token_cache",
    "set_token_profile",
    "set_oauth_profile",
]
