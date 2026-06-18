"""provider/openai_compat_helpers.py — OpenAI compat provider 的辅助常量与纯函数。"""
from __future__ import annotations

import json
from typing import Any

import httpx

# embed 输入字符上限（DashScope text-embedding-v3 单次最大约 6000 tokens，保守按字符计）
_EMBED_MAX_CHARS: int = 6000

# thinking level → budget_max 的比例
_LEVEL_FRACS: dict[str, float] = {
    "minimal": 0.05,
    "low": 0.15,
    "medium": 0.40,
    "high": 1.00,
}

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_EDITOR_VERSION = "vscode/1.96.2"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.7"
COPILOT_EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"
COPILOT_GITHUB_API_VERSION = "2025-04-01"
DEFAULT_COPILOT_API_BASE_URL = "https://api.individual.githubcopilot.com"

_MAX_COMPLETION_TOKENS_DEFAULT = 16384


def _copilot_reasoning_effort(level: str) -> str:
    return "low" if level == "minimal" else level


def _raise_for_status_with_body(resp: httpx.Response) -> None:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = (resp.text or "").strip().replace("\n", " ")
        if not body:
            raise
        raise httpx.HTTPStatusError(
            f"{exc} body={body}",
            request=exc.request,
            response=exc.response,
        ) from exc


def _extract_responses_text(data: dict[str, Any]) -> str:
    """从 responses API 的返回中提取文本。"""
    if isinstance(data.get("output_text"), str) and data.get("output_text"):
        return str(data["output_text"])
    output = data.get("output") or []
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
    return "\n".join(text_parts).strip()


def _extract_responses_stream_data(text: str) -> dict[str, Any]:
    """Parse responses API SSE text into a response-like dict."""
    deltas: list[str] = []
    completed: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "error":
            error = event.get("error") or event
            message = error.get("message") if isinstance(error, dict) else None
            raise RuntimeError(str(message or error))
        if isinstance(event.get("delta"), str) and event_type == "response.output_text.delta":
            deltas.append(str(event["delta"]))
        response = event.get("response")
        if isinstance(response, dict):
            if event_type == "response.completed":
                completed = response
            if isinstance(response.get("usage"), dict):
                usage = response["usage"]
    if completed is not None:
        output_text = _extract_responses_text(completed)
        if output_text:
            return {"output_text": output_text, "usage": completed.get("usage") or usage}
    return {"output_text": "".join(deltas).strip(), "usage": usage}


def _normalize_responses_content_part(part: dict[str, Any]) -> dict[str, Any]:
    part_type = str(part.get("type") or "")
    if part_type in {"text", "input_text"}:
        return {
            "type": "input_text",
            "text": str(part.get("text") or ""),
        }

    if part_type in {"image_url", "input_image"}:
        image_value = part.get("image_url")
        detail = str(part.get("detail") or "").strip()
        if isinstance(image_value, dict):
            detail = str(image_value.get("detail") or detail).strip()
            image_value = image_value.get("url")
        normalized: dict[str, Any] = {"type": "input_image"}
        if isinstance(image_value, str) and image_value.strip():
            normalized["image_url"] = image_value.strip()
        if detail:
            normalized["detail"] = detail
        return normalized if "image_url" in normalized else dict(part)

    return dict(part)


def _normalize_responses_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    normalized: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict):
            normalized.append(_normalize_responses_content_part(item))
    return normalized


def _build_copilot_ide_headers(*, include_api_version: bool = False) -> dict[str, str]:
    headers = {
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Editor-Plugin-Version": COPILOT_EDITOR_PLUGIN_VERSION,
        "User-Agent": COPILOT_USER_AGENT,
    }
    if include_api_version:
        headers["X-Github-Api-Version"] = COPILOT_GITHUB_API_VERSION
    return headers


def _resolve_copilot_proxy_host(proxy_ep: str) -> str | None:
    trimmed = proxy_ep.strip()
    if not trimmed:
        return None
    url_text = trimmed if trimmed.startswith(("http://", "https://")) else f"https://{trimmed}"
    try:
        parsed = httpx.URL(url_text)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    host = (parsed.host or "").strip().lower()
    return host or None


def _derive_copilot_api_base_url_from_token(token: str) -> str | None:
    trimmed = token.strip()
    if not trimmed:
        return None
    marker = "proxy-ep="
    for part in trimmed.split(";"):
        part = part.strip()
        if part.lower().startswith(marker):
            host = _resolve_copilot_proxy_host(part[len(marker):])
            if not host:
                return None
            return f"https://{host.replace('proxy.', 'api.', 1)}"
    return None


def _normalize_copilot_api_base_url(raw: str) -> str:
    trimmed = raw.strip().rstrip("/")
    if not trimmed or trimmed == "https://api.githubcopilot.com":
        return DEFAULT_COPILOT_API_BASE_URL
    return trimmed
