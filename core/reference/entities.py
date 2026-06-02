from __future__ import annotations

import time
from typing import Any

from .models import ResolvedEntity, ResolvedSpeaker
from .reasoning import reason_about_candidates_with_llm


async def resolve_entities(
    resolver: object,
    message: str,
    semantic: Any,
    episodic: Any,
) -> list[ResolvedEntity]:
    if not message or not message.strip():
        return []

    log = getattr(resolver, "_log", None)
    total_t0 = time.perf_counter()
    sigs = resolver.extract_signals(message)
    candidates_t0 = time.perf_counter()
    candidates = resolver._retrieve_candidates(message, sigs, semantic, episodic)
    if log is not None:
        log.info(
            "[reference.entities] candidates_ready dt=%.3fs candidates=%d topic_anchors=%d",
            time.perf_counter() - candidates_t0,
            len(candidates),
            len(getattr(sigs, "topic_anchors", []) or []),
        )
    if not candidates:
        if log is not None:
            log.info("[reference.entities] no_candidates total_dt=%.3fs", time.perf_counter() - total_t0)
        return []

    llm_results: list[dict[str, Any]] = []
    if resolver._provider is not None:
        llm_t0 = time.perf_counter()
        llm_results = await reason_about_candidates_with_llm(resolver, message, candidates)
        if log is not None:
            log.info(
                "[reference.entities] llm_reason_done dt=%.3fs results=%d",
                time.perf_counter() - llm_t0,
                len(llm_results),
            )

    entities: list[ResolvedEntity] = []

    if llm_results:
        for item in llm_results:
            nid = str(item.get("node_id", ""))
            if nid not in candidates:
                continue
            confidence = float(item.get("confidence", 0.0))
            if confidence < resolver._thresholds.reference_min_confidence:
                continue
            nd = candidates[nid]
            entities.append(
                ResolvedEntity(
                    node_id=nid,
                    title=nd.get("title", nid),
                    kind=nd.get("kind", "unknown"),
                    confidence=round(confidence, 2),
                    snippet=nd.get("body", ""),
                    signal_types=nd.get("_sig", []),
                    relationship_note=str(item.get("relationship_note", "")),
                )
            )
    else:
        for nid, nd in candidates.items():
            sigs_hit = nd.get("_sig", [])
            base = resolver._thresholds.reference_local_signal_base + len(set(sigs_hit)) * resolver._thresholds.reference_local_signal_step
            if base < resolver._thresholds.reference_min_confidence:
                continue
            entities.append(
                ResolvedEntity(
                    node_id=nid,
                    title=nd.get("title", nid),
                    kind=nd.get("kind", "unknown"),
                    confidence=round(min(base, resolver._thresholds.reference_local_confidence_cap), 2),
                    snippet=nd.get("body", ""),
                    signal_types=sigs_hit,
                    relationship_note="（本地评分，LLM 不可用）",
                )
            )

    entities.sort(key=lambda e: e.confidence, reverse=True)
    if log is not None:
        log.info(
            "[reference.entities] resolved total_dt=%.3fs entities=%d mode=%s",
            time.perf_counter() - total_t0,
            len(entities),
            "llm" if llm_results else "local",
        )
    return entities


def format_section(entities: list[ResolvedEntity]) -> str:
    if not entities:
        return "（无可链接的历史实体）"
    lines = ["从记忆中识别到以下相关实体（LLM 推理确认，按置信度排列）："]
    for entity in entities:
        note = f" — {entity.relationship_note}" if entity.relationship_note and "本地评分" not in entity.relationship_note else ""
        lines.append(f"- [{entity.kind}] {entity.title}（confidence:{entity.confidence:.2f}{note}）")
        if entity.snippet:
            lines.append(f"  {entity.snippet}")
    return "\n".join(lines)


def format_speaker_section(speaker: ResolvedSpeaker | None) -> str:
    if speaker is None:
        return "（当前轮尚未稳定识别当前交互对象，先依赖本轮消息与 chat 连续性）"
    status = "临时画像" if speaker.provisional else "稳定画像"
    lines = [f"当前交互对象候选: {speaker.title}（confidence:{speaker.confidence:.2f}，{status}）"]
    if speaker.relationship_note:
        lines.append(f"判断: {speaker.relationship_note}")
    lines.extend(f"- {item}" for item in speaker.evidence)
    if speaker.snippet:
        lines.append(f"画像记忆: {speaker.snippet}")
    return "\n".join(lines)
