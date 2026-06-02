from __future__ import annotations

from typing import Any

from .common import chat_handle_tag, normalize_text, split_text_sentences
from .extraction import extract_identity_cues
from .models import ExtractedSignals


_SPEAKER_CONTINUITY_MAX_QUERIES = 4
_SPEAKER_CONTINUITY_MAX_CHARS = 96


def _speaker_continuity_queries(
    chat_continuity: str,
    *,
    chat_id: str = "",
    source_hint: str = "",
) -> list[str]:
    text = normalize_text(chat_continuity)
    if not text:
        return []

    queries: list[str] = []

    def _add(query: str) -> None:
        normalized = normalize_text(query)
        if not normalized:
            return
        if len(normalized) > _SPEAKER_CONTINUITY_MAX_CHARS:
            normalized = normalized[-_SPEAKER_CONTINUITY_MAX_CHARS :].strip()
        if normalized and normalized not in queries:
            queries.append(normalized)

    continuity_cues = extract_identity_cues(text, chat_id=chat_id, source_hint=source_hint)
    for query in [
        *continuity_cues.get("names", []),
        *continuity_cues.get("preferences", []),
        *continuity_cues.get("explicit", []),
    ]:
        _add(query)
        if len(queries) >= _SPEAKER_CONTINUITY_MAX_QUERIES:
            return queries

    for sentence in reversed(split_text_sentences(text)):
        _add(sentence)
        if len(queries) >= _SPEAKER_CONTINUITY_MAX_QUERIES:
            break
    return queries


def retrieve_candidates(
    resolver: object,
    message: str,
    sigs: ExtractedSignals,
    semantic: Any,
    episodic: Any,
    source: str | None = None,
) -> dict[str, dict[str, Any]]:
    seen: set[str] = set()
    candidates: dict[str, dict[str, Any]] = {}

    def _add(nodes: list[dict[str, Any]], sig: str) -> None:
        for nd in nodes:
            nid = nd.get("id", "")
            if nid and nid not in seen:
                seen.add(nid)
                nd["_sig"] = nd.get("_sig", []) + [sig]
                candidates[nid] = nd

    anchors: list[str] = []
    for anchor in [message, *sigs.topic_anchors]:
        if anchor and anchor not in anchors:
            anchors.append(anchor)
        if len(anchors) >= resolver._thresholds.reference_max_anchors:
            break
    _add(semantic.retrieve_multi_anchor(anchors, top_k=resolver._thresholds.reference_topic_top_k, source=source), "topic")

    recent_rows = episodic.list_recent_narrative(limit=resolver._thresholds.reference_recent_narrative_limit)
    for row in recent_rows:
        content = row.get("content", "")
        if content:
            _add(semantic.retrieve(content, top_k=resolver._thresholds.reference_recent_semantic_top_k, source=source), "recent")

    return dict(candidates)


def retrieve_speaker_candidates(
    resolver: object,
    message: str,
    semantic: Any,
    *,
    chat_id: str = "",
    recent_turns: list[dict[str, Any]] | None = None,
    chat_continuity: str = "",
    cached_profile_id: str = "",
    source_hint: str = "",
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    cues = extract_identity_cues(message, chat_id=chat_id, source_hint=source_hint)
    candidates: dict[str, dict[str, Any]] = {}

    def _retrieve_profiles(query: str, *, top_k: int, tag: str | None = None) -> list[dict[str, Any]]:
        normalized = query.strip()
        if not normalized:
            return []
        nodes: list[dict[str, Any]] = []
        nodes.extend(semantic.retrieve(normalized, top_k=top_k, kind="interlocutor", tag=tag))
        return nodes

    def _add(nodes: list[dict[str, Any]], signal: str) -> None:
        for raw in nodes:
            node_id = str(raw.get("id") or "").strip()
            if not node_id:
                continue
            existing = candidates.get(node_id)
            if existing is None:
                item = dict(raw)
                item["_sig"] = [signal]
                candidates[node_id] = item
            else:
                sigs = list(existing.get("_sig") or [])
                if signal not in sigs:
                    sigs.append(signal)
                existing["_sig"] = sigs

    if cached_profile_id:
        cached = semantic.get(cached_profile_id)
        if cached is not None and cached.kind == "interlocutor":
            _add([cached.to_dict()], "cached")

    if message.strip():
        _add(_retrieve_profiles(message, top_k=3), "message")

    for name in cues["names"]:
        _add(_retrieve_profiles(name, top_k=3), "self_name")

    if chat_id:
        handle_tag = chat_handle_tag(chat_id)
        _add(_retrieve_profiles(chat_id, top_k=2, tag=handle_tag), "handle_tag")
        _add(_retrieve_profiles(chat_id, top_k=2), "handle_text")

    if chat_continuity.strip():
        continuity_queries = _speaker_continuity_queries(
            chat_continuity,
            chat_id=chat_id,
            source_hint=source_hint,
        )
        log = getattr(resolver, "_log", None)
        if log is not None:
            log.info(
                "[reference.speaker] continuity_queries=%d chat_continuity_chars=%d",
                len(continuity_queries),
                len(chat_continuity),
            )
        for query in continuity_queries:
            _add(_retrieve_profiles(query, top_k=1), "chat_continuity")

    for trait in cues.get("source_traits", []):
        _add(_retrieve_profiles(trait, top_k=1), "source_trait")

    for turn in (recent_turns or [])[-4:]:
        content = normalize_text(str(turn.get("content") or ""))
        if not content:
            continue
        _add(_retrieve_profiles(content, top_k=1), "recent_turn")

    return dict(candidates), cues
