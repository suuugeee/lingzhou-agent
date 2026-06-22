from __future__ import annotations

import time
from typing import Any, cast

from .common import chat_handle_tag, normalize_text, split_text_sentences
from .extraction import extract_identity_cues
from .models import ExtractedSignals

_SPEAKER_CONTINUITY_MAX_QUERIES = 4
_SPEAKER_CONTINUITY_MAX_CHARS = 96
_SPEAKER_LOG_QUERY_PREVIEW_CHARS = 48
_REFERENCE_OPERATIONAL_MEMORY_KINDS = frozenset({
    "execute_result",
    "meta_reflection",
    "run_monitor",
    "run_result",
    "task_progress",
    "working_trace",
})


def _speaker_identity_queries_from_cues(cues: dict[str, list[str]], *, max_queries: int = _SPEAKER_CONTINUITY_MAX_QUERIES) -> list[str]:
    queries: list[str] = []

    def _add(query: str) -> None:
        normalized = normalize_text(query)
        if not normalized:
            return
        if len(normalized) > _SPEAKER_CONTINUITY_MAX_CHARS:
            normalized = normalized[-_SPEAKER_CONTINUITY_MAX_CHARS :].strip()
        if normalized and normalized not in queries:
            queries.append(normalized)

    for query in [
        *cues.get("names", []),
        *cues.get("preferences", []),
        *cues.get("explicit", []),
    ]:
        _add(query)
        if len(queries) >= max_queries:
            break
    return queries


def _speaker_query_preview(query: str) -> str:
    normalized = normalize_text(query)
    if len(normalized) <= _SPEAKER_LOG_QUERY_PREVIEW_CHARS:
        return normalized
    return normalized[:_SPEAKER_LOG_QUERY_PREVIEW_CHARS].rstrip() + " ..."


def _speaker_continuity_queries(
    chat_continuity: str,
    *,
    chat_id: str = "",
    source_hint: str = "",
) -> list[str]:
    text = normalize_text(chat_continuity)
    if not text:
        return []

    continuity_cues = extract_identity_cues(text, chat_id=chat_id, source_hint=source_hint)
    queries = _speaker_identity_queries_from_cues(continuity_cues)
    if len(queries) >= _SPEAKER_CONTINUITY_MAX_QUERIES:
        return queries

    def _add(query: str) -> None:
        normalized = normalize_text(query)
        if not normalized:
            return
        if len(normalized) > _SPEAKER_CONTINUITY_MAX_CHARS:
            normalized = normalized[-_SPEAKER_CONTINUITY_MAX_CHARS :].strip()
        if normalized and normalized not in queries:
            queries.append(normalized)

    for sentence in reversed(split_text_sentences(text)):
        _add(sentence)
        if len(queries) >= _SPEAKER_CONTINUITY_MAX_QUERIES:
            break
    return queries


def _retrieve_profiles_by_exact_tag(
    semantic: Any,
    tag: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    normalized_tag = str(tag or "").strip()
    if not normalized_tag:
        return []

    db_session = cast(Any, getattr(semantic, "_db_session", None))
    load_filtered = cast(Any, getattr(semantic, "_load_filtered", None))
    if callable(db_session) and callable(load_filtered):
        try:
            with cast(Any, db_session)():
                nodes = cast(list[Any], load_filtered(kind="interlocutor", tag=normalized_tag, limit=top_k))
            return [cast(Any, node).to_dict() for node in nodes]
        except Exception:
            pass

    return list(semantic.retrieve(normalized_tag, top_k=top_k, kind="interlocutor", tag=normalized_tag))


def _retrieve_profiles_by_name(
    semantic: Any,
    name: str,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    normalized = normalize_text(name)
    if not normalized:
        return []
    nodes = _retrieve_profiles_by_exact_tag(semantic, f"alias:{normalized}", top_k=top_k)
    if nodes:
        return nodes
    return list(semantic.retrieve(normalized, top_k=top_k, kind="interlocutor"))


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
            if str(nd.get("kind") or "").strip() in _REFERENCE_OPERATIONAL_MEMORY_KINDS:
                continue
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
    identity_queries = _speaker_identity_queries_from_cues(cues)
    candidates: dict[str, dict[str, Any]] = {}
    log = getattr(resolver, "_log", None)

    def _retrieve_profiles(query: str, *, top_k: int, tag: str | None = None) -> list[dict[str, Any]]:
        normalized = query.strip()
        if not normalized:
            return []
        nodes: list[dict[str, Any]] = []
        nodes.extend(semantic.retrieve(normalized, top_k=top_k, kind="interlocutor", tag=tag))
        return nodes

    def _lookup_query(
        query: str,
        *,
        top_k: int,
        signal: str,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        lookup_t0 = time.perf_counter()
        nodes = _retrieve_profiles(query, top_k=top_k, tag=tag)
        if log is not None:
            log.info(
                "[reference.speaker] lookup signal=%s dt=%.3fs candidates=%d chars=%d preview=%s",
                signal,
                time.perf_counter() - lookup_t0,
                len(nodes),
                len(query),
                _speaker_query_preview(query),
            )
        return nodes

    def _lookup_name(name: str, *, top_k: int, signal: str) -> list[dict[str, Any]]:
        lookup_t0 = time.perf_counter()
        nodes = _retrieve_profiles_by_name(semantic, name, top_k=top_k)
        if log is not None:
            log.info(
                "[reference.speaker] lookup signal=%s dt=%.3fs candidates=%d chars=%d preview=%s",
                signal,
                time.perf_counter() - lookup_t0,
                len(nodes),
                len(name),
                _speaker_query_preview(name),
            )
        return nodes

    def _lookup_exact_tag(tag: str, *, top_k: int, signal: str) -> list[dict[str, Any]]:
        lookup_t0 = time.perf_counter()
        nodes = _retrieve_profiles_by_exact_tag(semantic, tag, top_k=top_k)
        if log is not None:
            log.info(
                "[reference.speaker] lookup signal=%s dt=%.3fs candidates=%d chars=%d preview=%s",
                signal,
                time.perf_counter() - lookup_t0,
                len(nodes),
                len(tag),
                _speaker_query_preview(tag),
            )
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

    if candidates and not identity_queries:
        if log is not None:
            log.info(
                "[reference.speaker] cached_short_circuit candidates=%d source_traits=%d recent_turns=%d chat_continuity_chars=%d",
                len(candidates),
                len(cues.get("source_traits", [])),
                len((recent_turns or [])[-4:]),
                len(chat_continuity),
            )
        return dict(candidates), cues

    for name in cues["names"]:
        _add(_lookup_name(name, top_k=3, signal="self_name"), "self_name")

    for query in identity_queries:
        _add(_lookup_query(query, top_k=1, signal="message_identity"), "message_identity")

    if chat_id:
        handle_tag = chat_handle_tag(chat_id)
        handle_t0 = time.perf_counter()
        handle_nodes = _retrieve_profiles_by_exact_tag(semantic, handle_tag, top_k=2)
        if log is not None:
            log.info(
                "[reference.speaker] handle_lookup dt=%.3fs candidates=%d chat_id_chars=%d",
                time.perf_counter() - handle_t0,
                len(handle_nodes),
                len(chat_id),
            )
        _add(handle_nodes, "handle_tag")

    if chat_continuity.strip():
        continuity_queries = _speaker_continuity_queries(
            chat_continuity,
            chat_id=chat_id,
            source_hint=source_hint,
        )
        if log is not None:
            log.info(
                "[reference.speaker] continuity_queries=%d chat_continuity_chars=%d",
                len(continuity_queries),
                len(chat_continuity),
            )
        for query in continuity_queries:
            if query in cues.get("names", []):
                _add(_lookup_name(query, top_k=1, signal="chat_continuity"), "chat_continuity")
            else:
                _add(_lookup_query(query, top_k=1, signal="chat_continuity"), "chat_continuity")

    for trait in cues.get("source_traits", []):
        _add(_lookup_exact_tag(trait, top_k=1, signal="source_trait"), "source_trait")

    for turn in (recent_turns or [])[-4:]:
        content = normalize_text(str(turn.get("content") or ""))
        if not content:
            continue
        turn_cues = extract_identity_cues(content, chat_id=chat_id, source_hint=source_hint)
        for name in turn_cues.get("names", []):
            _add(_lookup_name(name, top_k=1, signal="recent_turn"), "recent_turn")
        for query in [*turn_cues.get("preferences", []), *turn_cues.get("explicit", [])]:
            _add(_lookup_query(query, top_k=1, signal="recent_turn"), "recent_turn")

    return dict(candidates), cues
