from __future__ import annotations

import time
from typing import Any

from core.metabolic import StateProposal

from .common import chat_handle_tag, default_interlocutor_title, normalize_text, short_text_digest
from .extraction import extract_identity_cues
from .models import ResolvedSpeaker
from .reasoning import reason_about_speaker_with_llm
from .retrieval import retrieve_speaker_candidates


def resolve_speaker_locally(
    resolver: object,
    candidates: dict[str, dict[str, Any]],
    *,
    cues: dict[str, list[str]],
    chat_id: str = "",
    cached_profile_id: str = "",
) -> ResolvedSpeaker | None:
    best: tuple[float, dict[str, Any]] | None = None
    lowered_names = [name.lower() for name in cues.get("names", [])]
    handle_tag = chat_handle_tag(chat_id) if chat_id else ""
    for node_id, node in candidates.items():
        score = float(node.get("score") or 0.0)
        signal_types = list(node.get("_sig") or [])
        score += len(set(signal_types)) * 0.12
        title_body = f"{node.get('title', '')} {node.get('body', '')}".lower()
        tags = {str(tag) for tag in (node.get("tags") or [])}
        if cached_profile_id and node_id == cached_profile_id:
            score += 0.22
        if lowered_names and any(name and name in title_body for name in lowered_names):
            score += 0.35
        if chat_id and (handle_tag in tags or chat_id.lower() in title_body):
            score += 0.26
        if cues.get("source_traits") and any(trait.split("=", 1)[-1] in title_body for trait in cues.get("source_traits", [])):
            score += 0.18
        if best is None or score > best[0]:
            best = (score, node)

    if best is None:
        return None
    confidence = min(best[0], resolver._thresholds.reference_local_confidence_cap)
    if confidence < resolver._thresholds.reference_min_confidence:
        return None
    node = best[1]
    return ResolvedSpeaker(
        node_id=str(node.get("id") or ""),
        title=str(node.get("title") or default_interlocutor_title(chat_id)),
        confidence=round(confidence, 2),
        snippet=str(node.get("body") or ""),
        evidence=[f"本地多线索命中：{', '.join(node.get('_sig', []))}"] if node.get("_sig") else [],
        relationship_note="多线索画像匹配",
        signal_types=list(node.get("_sig") or []),
        provisional=False,
        search_anchors=[str(node.get("title") or "")],
        source_traits=list(cues.get("source_traits") or []),
    )


def build_provisional_speaker(
    message: str,
    *,
    cues: dict[str, list[str]],
    chat_id: str = "",
    hint_title: str = "",
) -> ResolvedSpeaker | None:
    display_name = (cues.get("names") or [])[0] if cues.get("names") else hint_title.strip()
    if not display_name:
        display_name = default_interlocutor_title(chat_id)
    if not display_name and not chat_id:
        return None
    seed = "|".join(part for part in (chat_id, display_name, message) if part)
    node_id = f"interlocutor-profile-{short_text_digest(seed or message or display_name)}"
    evidence: list[str] = []
    if cues.get("names"):
        evidence.append(f"当前消息出现自称：{cues['names'][0]}")
    if cues.get("preferences"):
        evidence.append(f"当前消息出现偏好：{cues['preferences'][0]}")
    if chat_id:
        evidence.append("当前 chat 只能提供延续线索，不能单独定身份")
    snippet_parts = [item for item in [*cues.get("preferences", []), *cues.get("explicit", [])] if item]
    confidence = 0.72 if cues.get("names") else 0.46 if chat_id else 0.38
    return ResolvedSpeaker(
        node_id=node_id,
        title=display_name,
        confidence=round(confidence, 2),
        snippet="；".join(snippet_parts) or message,
        evidence=evidence,
        relationship_note="当前轮形成临时交互对象画像",
        signal_types=["self_intro" if cues.get("names") else "provisional"],
        provisional=True,
        search_anchors=[display_name, *cues.get("preferences", [])],
        source_traits=list(cues.get("source_traits") or []),
    )


async def resolve_current_speaker(
    resolver: object,
    message: str,
    semantic: Any,
    *,
    chat_id: str = "",
    recent_turns: list[dict[str, Any]] | None = None,
    chat_continuity: str = "",
    interlocutor_continuity: str = "",
    cached_profile_id: str = "",
    source_hint: str = "",
) -> ResolvedSpeaker | None:
    if not normalize_text(message):
        return None

    log = getattr(resolver, "_log", None)
    total_t0 = time.perf_counter()
    candidates_t0 = time.perf_counter()
    candidates, cues = retrieve_speaker_candidates(
        resolver,
        message,
        semantic,
        chat_id=chat_id,
        recent_turns=recent_turns,
        chat_continuity=chat_continuity,
        source_hint=source_hint,
        cached_profile_id=cached_profile_id,
    )
    if log is not None:
        log.info(
            "[reference.speaker] candidates_ready dt=%.3fs candidates=%d names=%d preferences=%d explicit=%d source_traits=%d recent_turns=%d chat_continuity_chars=%d cached_profile=%s",
            time.perf_counter() - candidates_t0,
            len(candidates),
            len(cues.get("names", [])),
            len(cues.get("preferences", [])),
            len(cues.get("explicit", [])),
            len(cues.get("source_traits", [])),
            len((recent_turns or [])[-4:]),
            len(chat_continuity),
            cached_profile_id or "",
        )

    llm_result: dict[str, Any] = {}
    if resolver._provider is not None and (candidates or any(cues.values()) or chat_continuity.strip() or interlocutor_continuity.strip()):
        llm_t0 = time.perf_counter()
        llm_result = await reason_about_speaker_with_llm(
            resolver,
            message,
            candidates=candidates,
            recent_turns=recent_turns,
            chat_continuity=chat_continuity,
            interlocutor_continuity=interlocutor_continuity,
            chat_id=chat_id,
            source_hint=source_hint,
            cues=cues,
        )
        if log is not None:
            log.info(
                "[reference.speaker] llm_reason_done dt=%.3fs result_keys=%s",
                time.perf_counter() - llm_t0,
                sorted(llm_result.keys()),
            )

    if llm_result:
        node_id = str(llm_result.get("node_id") or "").strip()
        confidence = round(float(llm_result.get("confidence") or 0.0), 2)
        evidence = [str(item).strip() for item in (llm_result.get("evidence") or []) if str(item).strip()]
        note = str(llm_result.get("relationship_note") or "").strip()
        if node_id in candidates and confidence >= resolver._thresholds.reference_min_confidence:
            node = candidates[node_id]
            title = str(llm_result.get("display_name") or node.get("title") or default_interlocutor_title(chat_id)).strip()
            anchors = [title, *cues.get("names", []), *cues.get("preferences", [])]
            return ResolvedSpeaker(
                node_id=node_id,
                title=title,
                confidence=confidence,
                snippet=str(node.get("body") or ""),
                evidence=evidence,
                relationship_note=note,
                signal_types=list(node.get("_sig") or []),
                provisional=bool(llm_result.get("provisional")),
                search_anchors=[anchor for anchor in anchors if anchor],
                source_traits=list(cues.get("source_traits") or []),
            )
        if node_id == "NEW":
            provisional = build_provisional_speaker(
                message,
                cues=cues,
                chat_id=chat_id,
                hint_title=str(llm_result.get("display_name") or "").strip(),
            )
            if provisional is not None:
                provisional.confidence = max(provisional.confidence, confidence or provisional.confidence)
                if evidence:
                    provisional.evidence = evidence
                if note:
                    provisional.relationship_note = note
                if log is not None:
                    log.info(
                        "[reference.speaker] resolved total_dt=%.3fs mode=llm_new node=%s confidence=%.2f",
                        time.perf_counter() - total_t0,
                        provisional.node_id,
                        provisional.confidence,
                    )
            return provisional
        if node_id == "UNKNOWN":
            if log is not None:
                log.info("[reference.speaker] resolved total_dt=%.3fs mode=llm_unknown", time.perf_counter() - total_t0)
            return None

    resolved = resolve_speaker_locally(
        resolver,
        candidates,
        cues=cues,
        chat_id=chat_id,
        cached_profile_id=cached_profile_id,
    )
    if resolved is not None:
        if log is not None:
            log.info(
                "[reference.speaker] resolved total_dt=%.3fs mode=local node=%s confidence=%.2f",
                time.perf_counter() - total_t0,
                resolved.node_id,
                resolved.confidence,
            )
        return resolved
    provisional = build_provisional_speaker(message, cues=cues, chat_id=chat_id)
    if log is not None:
        log.info(
            "[reference.speaker] resolved total_dt=%.3fs mode=%s node=%s confidence=%.2f",
            time.perf_counter() - total_t0,
            "provisional" if provisional is not None else "none",
            getattr(provisional, "node_id", "") if provisional is not None else "",
            float(getattr(provisional, "confidence", 0.0) or 0.0),
        )
    return provisional


async def remember_speaker(
    speaker: ResolvedSpeaker,
    resolver: object,
    semantic: Any,
    task_store: Any | None,
    *,
    message: str,
    chat_id: str = "",
    task_id: str | int | None = None,
    source_hint: str = "",
    metabolic: Any | None = None,
) -> None:
    if not speaker.node_id:
        return

    from datetime import UTC, datetime

    from store.semantic import MemoryNode

    cues = extract_identity_cues(message, chat_id=chat_id, source_hint=source_hint)
    existing = semantic.get(speaker.node_id)
    merged_lines: list[str] = []
    if existing is not None and existing.body.strip():
        merged_lines.extend([line.strip() for line in existing.body.splitlines() if line.strip()])
    additions = [
        f"画像摘要: {speaker.snippet}" if speaker.snippet else "",
        f"识别判断: {speaker.relationship_note}" if speaker.relationship_note else "",
        *[f"识别依据: {item}" for item in speaker.evidence],
        *[f"偏好线索: {item}" for item in cues.get("preferences", [])],
        *[f"显式记忆要求: {item}" for item in cues.get("explicit", [])],
        *[f"来源特征: {item}" for item in cues.get("source_traits", [])],
        *([f"已见 chat 线索: {chat_id}"] if chat_id else []),
    ]
    for line in additions:
        normalized = normalize_text(line)
        if normalized and normalized not in merged_lines:
            merged_lines.append(normalized)

    tags = set(existing.tags if existing is not None else [])
    tags.update({"interlocutor_profile", f"interlocutor:{speaker.node_id}"})
    if chat_id:
        tags.update({f"chat:{chat_id}", chat_handle_tag(chat_id)})
    for alias in cues.get("names", []):
        tags.add(f"alias:{alias}")
    for trait in cues.get("source_traits", []):
        tags.add(trait)

    semantic.upsert(
        MemoryNode(
            id=speaker.node_id,
            kind="interlocutor",
            title=speaker.title or (existing.title if existing is not None else default_interlocutor_title(chat_id)),
            body="\n".join(merged_lines[-12:]),
            activation=max(existing.activation if existing is not None else 0.0, max(0.55, speaker.confidence)),
            importance=max(existing.importance if existing is not None else 0.0, 0.58 if not speaker.provisional else 0.45),
            valence=existing.valence if existing is not None else 0.5,
            tags=sorted(tags),
            source=(existing.source if existing is not None and existing.source else "interlocutor_profile"),
            created_at=existing.created_at if existing is not None else datetime.now(UTC).isoformat(),
        )
    )

    if task_store is None:
        return
    if metabolic is None:
        from core.metabolic import MetabolicEngine

        metabolic = MetabolicEngine(task_store)
    if chat_id:
        await metabolic.submit(
            StateProposal(
                op="set_fact",
                key=f"chat:{chat_id}:interlocutor_profile_id",
                value=speaker.node_id,
                scope="profile",
                source="reference/speaker",
            )
        )
    if task_id is not None:
        await metabolic.submit(
            StateProposal(
                op="set_fact",
                key=f"task:{task_id}:interlocutor_profile_id",
                value=speaker.node_id,
                scope="profile",
                source="reference/speaker",
            )
        )
    await metabolic.submit(
        StateProposal(
            op="set_fact",
            key=f"interlocutor:{speaker.node_id}:display_name",
            value=speaker.title,
            scope="profile",
            source="reference/speaker",
        )
    )
    if chat_id:
        await metabolic.submit(
            StateProposal(
                op="set_fact",
                key=f"interlocutor:{speaker.node_id}:handle:{short_text_digest(chat_id)}",
                value=chat_id,
                scope="profile",
                source="reference/speaker",
            )
        )
    for pref in cues.get("preferences", []):
        await metabolic.submit(
            StateProposal(
                op="set_fact",
                key=f"interlocutor:{speaker.node_id}:preference:{short_text_digest(pref)}",
                value=pref,
                scope="profile",
                source="reference/speaker",
            )
        )
    for explicit in cues.get("explicit", []):
        await metabolic.submit(
            StateProposal(
                op="set_fact",
                key=f"interlocutor:{speaker.node_id}:explicit:{short_text_digest(explicit)}",
                value=explicit,
                scope="profile",
                source="reference/speaker",
            )
        )
    for trait in cues.get("source_traits", []):
        await metabolic.submit(
            StateProposal(
                op="set_fact",
                key=f"interlocutor:{speaker.node_id}:source_trait:{short_text_digest(trait)}",
                value=trait,
                scope="profile",
                source="reference/speaker",
            )
        )
