from __future__ import annotations

import json
from typing import Any

from .common import _REASON_SYSTEM, _SPEAKER_REASON_SYSTEM, normalize_text


def categorize_llm_error_code(err_text: str) -> str:
    text = (err_text or "").lower()
    if " 413 " in f" {text} " or "request entity too large" in text or "payload too large" in text:
        return "413"
    if " 429 " in f" {text} " or "too many requests" in text:
        return "429"
    if " 401 " in f" {text} " or "unauthorized" in text:
        return "401"
    if " 403 " in f" {text} " or "forbidden" in text:
        return "403"
    if " 400 " in f" {text} " or "bad request" in text:
        return "400"
    if "readtimeout" in text or "timeout" in text:
        return "timeout"
    return "other"


async def reason_about_candidates_with_llm(
    resolver: object,
    message: str,
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if resolver._provider is None:
        resolver._last_llm_error = ""
        resolver._last_llm_error_code = ""
        return []
    from provider.base import Message as LLMMessage

    cand_lines: list[str] = []
    for nid, nd in candidates.items():
        body_snippet = str(nd.get("body", "")).replace("\n", " ")
        created_at = str(nd.get("created_at", ""))
        cand_lines.append(
            f'  {{"id":"{nid}","kind":"{nd.get("kind","")}","title":"{nd.get("title","")}","created_at":"{created_at}","body":"{body_snippet}"}}'
        )
    cand_block = "[\n" + ",\n".join(cand_lines) + "\n]"

    user_content = f'用户消息："{message}"\n\n候选节点：\n{cand_block}'

    try:
        raw = await resolver._provider.chat(
            [
                LLMMessage(role="system", content=_REASON_SYSTEM),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=resolver._reason_temperature,
        )
    except Exception as exc:
        err_text = str(exc) or repr(exc)
        resolver._last_llm_error = err_text
        resolver._last_llm_error_code = categorize_llm_error_code(err_text)
        resolver._log.warning("[reference] LLM 推理失败，降级为本地评分: %s", exc)
        return []
    resolver._last_llm_error = ""
    resolver._last_llm_error_code = ""

    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []


async def reason_about_speaker_with_llm(
    resolver: object,
    message: str,
    *,
    candidates: dict[str, dict[str, Any]],
    recent_turns: list[dict[str, Any]] | None = None,
    chat_continuity: str = "",
    interlocutor_continuity: str = "",
    chat_id: str = "",
    source_hint: str = "",
    cues: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if resolver._provider is None:
        resolver._last_llm_error = ""
        resolver._last_llm_error_code = ""
        return {}
    from provider.base import Message as LLMMessage

    cues = cues or {"names": [], "preferences": [], "explicit": []}
    candidate_lines: list[str] = []
    for node_id, node in candidates.items():
        body_snippet = str(node.get("body") or "").replace("\n", " ")
        candidate_lines.append(
            json.dumps(
                {
                    "id": node_id,
                    "title": node.get("title", ""),
                    "tags": node.get("tags", []),
                    "created_at": node.get("created_at", ""),
                    "body": body_snippet,
                    "signals": node.get("_sig", []),
                },
                ensure_ascii=False,
            )
        )
    turns_block = []
    for turn in (recent_turns or [])[-4:]:
        role = str(turn.get("role") or "?")
        content = normalize_text(str(turn.get("content") or ""))
        if content:
            turns_block.append(f"- {role}: {content}")

    user_content = "\n".join(
        [
            f'当前用户消息："{message}"',
            f"当前 chat 句柄线索：{chat_id or '（无）'}",
            f"来源路由线索：{source_hint or '（无）'}",
            "从当前消息提取的线索：",
            f"- names: {cues.get('names', [])}",
            f"- preferences: {cues.get('preferences', [])}",
            f"- explicit: {cues.get('explicit', [])}",
            f"- source_traits: {cues.get('source_traits', [])}",
            "最近交互片段：",
            "\n".join(turns_block) if turns_block else "（无）",
            "当前 chat 连续性：",
            chat_continuity or "（无）",
            "当前对象跨 chat 交互连续性：",
            interlocutor_continuity or "（无）",
            "候选交互对象画像：",
            "[\n" + ",\n".join(candidate_lines) + "\n]" if candidate_lines else "[]",
        ]
    )

    try:
        raw = await resolver._provider.chat(
            [
                LLMMessage(role="system", content=_SPEAKER_REASON_SYSTEM),
                LLMMessage(role="user", content=user_content),
            ],
            temperature=resolver._reason_temperature,
        )
    except Exception as exc:
        err_text = str(exc) or repr(exc)
        resolver._last_llm_error = err_text
        resolver._last_llm_error_code = categorize_llm_error_code(err_text)
        resolver._log.warning("[reference] 当前说话人识别失败，降级为本地评分: %s", exc)
        return {}

    resolver._last_llm_error = ""
    resolver._last_llm_error_code = ""
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
