from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from typing import TYPE_CHECKING, Any, cast

from store.task import RUNNABLE_TASK_STATUSES

from ..context.facts import (
    _load_context_facts_snapshot,
    _load_durable_failure_snapshot,
)
from ..context.sections import (
    _fmt_config_snapshot,
    _fmt_interlocutor_continuity,
    _fmt_memory_recall,
    _fmt_memory_system,
    _fmt_soul,
)
from ..context.skills import (
    _fmt_primary_skill,
    _fmt_skill_catalog,
    _fmt_skills,
)
from ..context.tasks import _fmt_evolution_breakers
from .sections import (
    _build_context_memory_sections,
    _build_context_state_sections,
    _build_context_task_sections,
    _finalize_context_text,
)

if TYPE_CHECKING:
    from core.judgment.frame import CognitionFrame
    from core.perception import (
        CognitiveSignals,
        EmotionState,
        EthosState,
        JudgmentSignals,
        Percept,
        PerceptionReplaySummary,
    )
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore


_log = logging.getLogger("lingzhou.judgment")
_CROSS_TASK_EPISODIC_MAX_CHARS = 4000


def _clip_context_artifact(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def _build_context_anchors(
    assembler: Any,
    task: Any | None,
    user_message: str,
    resolved_chat_id: str,
    resolved_speaker: Any | None,
    failures: list[Any],
) -> list[str]:
    anchors: list[str] = []
    if task:
        primary_anchor = task.next_step or task.goal or task.title
        if primary_anchor:
            anchors.append(primary_anchor)
        task_source = str(getattr(task, "source", "") or "")
        if task_source and task_source not in anchors:
            anchors.append(task_source)
    if user_message and user_message not in anchors:
        anchors.append(user_message)
    if resolved_chat_id:
        chat_anchor = f"chat:{str(resolved_chat_id or '').strip()}"
        if chat_anchor not in anchors:
            anchors.append(chat_anchor)
    if resolved_speaker is not None:
        for anchor in [resolved_speaker.title, *resolved_speaker.search_anchors, f"interlocutor:{resolved_speaker.node_id}"]:
            normalized_anchor = str(anchor or "").strip()
            if normalized_anchor and normalized_anchor not in anchors:
                anchors.append(normalized_anchor)
    if failures:
        anchors.append(failures[0].kind)
    return anchors


async def _resolve_context_scope(
    assembler: Any,
    task_store: TaskStore,
    active_task: Any | None,
    user_message: str,
    chat_id: str | None,
) -> tuple[Any | None, bool, str | None, str, str]:
    task = active_task
    if task is None:
        normalized_chat_id = str(chat_id or "").strip()
        fact_keys = [f"focus:chat:{normalized_chat_id}"] if normalized_chat_id else []
        fact_keys.append("focus:current_task_id")
        for fact_key in fact_keys:
            try:
                raw_task_id, found = await task_store.get_fact(fact_key)
            except Exception:
                raw_task_id, found = "", False
            if not found:
                continue
            try:
                task_id = int(raw_task_id)
            except (TypeError, ValueError):
                continue
            if task_id <= 0:
                continue
            try:
                task = await task_store.get_task_by_id(task_id)
            except Exception:
                task = None
            if task is not None:
                break
    include_open_task_overview = task is None
    task_id_str = str(task.id) if task else None
    search_query = (user_message or (task.next_step or task.goal or task.title)) if task else user_message
    resolved_chat_id = str(chat_id or "").strip()
    if not resolved_chat_id and task is not None:
        try:
            value, found = await task_store.get_fact(f"task:{task.id}:chat_id")
        except Exception:
            value, found = "", False
        if found and str(value or "").strip():
            resolved_chat_id = str(value).strip()
        else:
            source = str(getattr(task, "source", "") or "").strip()
            if source.startswith("chat:"):
                resolved_chat_id = source[5:].strip()
    return task, include_open_task_overview, task_id_str, search_query, resolved_chat_id


async def _load_context_artifacts(
    assembler: Any,
    *,
    task_store: TaskStore,
    task: Any | None,
    include_open_task_overview: bool,
    search_query: str,
    resolved_chat_id: str,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    task_id_str = str(task.id) if task else None
    gather_t0 = time.perf_counter()
    futures: list[tuple[str, Any]] = [
        ("episodic_text", loop.run_in_executor(None, episodic.load_for_context, task_id_str, assembler._cfg.memory.episodic_n_recent)),
        ("chat_continuity", loop.run_in_executor(None, episodic.load_for_chat_context, resolved_chat_id, assembler._cfg.memory.episodic_n_recent) if resolved_chat_id else None),
        ("recent_turns", loop.run_in_executor(None, functools.partial(episodic.get_recent_turns, task_id_str, assembler._cfg.thresholds.chat_history_turn_limit, chat_id=resolved_chat_id)) if (resolved_chat_id or task_id_str) else None),
        ("chat_memories", loop.run_in_executor(None, functools.partial(semantic.retrieve, search_query or resolved_chat_id or "", min(3, assembler._cfg.memory.semantic_top_k), tag=f"chat:{str(resolved_chat_id or '').strip()}", source="chat_summary")) if resolved_chat_id else None),
        ("speaker_hint", asyncio.create_task(task_store.get_fact(f"chat:{resolved_chat_id}:interlocutor_profile_id")) if resolved_chat_id else None),
        ("recent_runs", asyncio.create_task(task_store.list_runs(task_id=task.id, limit=6)) if task else None),
        ("waiting_tasks", asyncio.create_task(task_store.list_tasks(status="waiting", limit=5)) if include_open_task_overview else None),
        ("durable_failure_snapshot", asyncio.create_task(_load_durable_failure_snapshot(task_store))),
        ("context_facts", asyncio.create_task(_load_context_facts_snapshot(task_store, task, exclude_prefixes=assembler._cfg.thresholds.fact_context_exclude_prefixes, task_limit=assembler._cfg.thresholds.fact_context_task_limit, global_limit=assembler._cfg.thresholds.fact_context_global_limit, priority_prefixes=assembler._cfg.thresholds.fact_context_priority_prefixes, priority_limit=assembler._cfg.thresholds.fact_context_priority_limit, recent_scan_multiplier=assembler._cfg.thresholds.fact_context_recent_scan_multiplier, recent_scan_min=assembler._cfg.thresholds.fact_context_recent_scan_min))),
        ("probes", asyncio.create_task(assembler._probe_manager.list_probes()) if assembler._probe_manager else None),
        ("failures", asyncio.create_task(task_store.list_failures_for_task(str(task.id), assembler._cfg.memory.failure_limit) if task else task_store.list_failures(assembler._cfg.memory.failure_limit))),
    ]
    if include_open_task_overview:
        list_runnable: Any = getattr(task_store, "list_runnable_tasks", None)
        if list_runnable is not None:
            futures.append(("runnable_tasks", asyncio.create_task(list_runnable(limit=8))))
        else:
            list_tasks: Any = getattr(task_store, "list_tasks", None)
            if list_tasks is not None:

                async def _load_runnable_tasks() -> list[Any]:
                    tasks = await list_tasks(limit=8)
                    return [item for item in tasks if getattr(item, "status", "") in RUNNABLE_TASK_STATUSES][:8]

                futures.append(("runnable_tasks", asyncio.create_task(_load_runnable_tasks())))

    active_futures = [(name, awaitable) for name, awaitable in futures if awaitable is not None]
    results = await asyncio.gather(*(awaitable for _, awaitable in active_futures), return_exceptions=True)
    data: dict[str, Any] = {}
    error: BaseException | None = None
    for (name, _), value in zip(active_futures, results, strict=False):
        if isinstance(value, BaseException):
            error = error or value
            continue
        data[name] = value
    if error is not None:
        raise error
    if data.get("chat_continuity", "").strip() == data.get("episodic_text", "").strip():
        data["chat_continuity"] = ""
    _log.info(
        "[context] base_artifacts_ready dt=%.3fs task=%s chat=%s episodic_chars=%d chat_continuity_chars=%d recent_turns=%d chat_memories=%d",
        time.perf_counter() - gather_t0,
        task_id_str or "",
        resolved_chat_id or "",
        len(str(data.get("episodic_text") or "")),
        len(str(data.get("chat_continuity") or "")),
        len(data.get("recent_turns") or []),
        len(data.get("chat_memories") or []),
    )
    similar_tasks: list[Any] = []
    if include_open_task_overview and str(search_query or "").strip():
        finder: Any = getattr(task_store, "find_similar_open_tasks", None)
        if finder is not None:
            exclude_task_ids = [task.id] if task is not None else None
            active_source = str(getattr(task, "source", "") or "").strip()
            similar_tasks = await finder(
                search_query,
                limit=5,
                min_score=assembler._cfg.thresholds.task_similarity_context_score,
                exclude_task_ids=exclude_task_ids,
                allowed_sources=("self_drive",) if active_source == "self_drive" else None,
                excluded_sources=None if active_source == "self_drive" else ("self_drive",),
            )
    cross_task_t0 = time.perf_counter()
    cross_task_episodic_text = ""
    if task_id_str and search_query:
        raw_cross_task_episodic_text = await loop.run_in_executor(
            None,
            episodic.search,
            search_query,
            _CROSS_TASK_EPISODIC_MAX_CHARS,
            task_id_str,
        )
        cross_task_episodic_text = _clip_context_artifact(
            raw_cross_task_episodic_text,
            _CROSS_TASK_EPISODIC_MAX_CHARS,
        )
        if len(str(raw_cross_task_episodic_text or "")) > len(cross_task_episodic_text):
            _log.warning(
                "[context] episodic_cross_task_clipped raw_chars=%d max_chars=%d",
                len(str(raw_cross_task_episodic_text or "")),
                _CROSS_TASK_EPISODIC_MAX_CHARS,
            )
    _log.info(
        "[context] episodic search=%r cross_task_hit=%s cross_task_chars=%d dt=%.3fs",
        (search_query or ""),
        bool(cross_task_episodic_text),
        len(cross_task_episodic_text),
        time.perf_counter() - cross_task_t0,
    )
    return {"data": data, "similar_tasks": similar_tasks, "cross_task_episodic_text": cross_task_episodic_text}


async def _resolve_context_references(
    assembler: Any,
    *,
    user_message: str,
    semantic: SemanticMemory,
    episodic: EpisodicMemory,
    task_store: TaskStore,
    task: Any | None,
    resolved_chat_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    refs_t0 = time.perf_counter()
    recent_turns = data.get("recent_turns", []) or []
    chat_continuity_text = str(data.get("chat_continuity", "") or "")
    _log.info(
        "[context.refs] start task=%s chat=%s user_chars=%d recent_turns=%d chat_continuity_chars=%d",
        str(task.id) if task else "",
        resolved_chat_id or "",
        len(user_message or ""),
        len(recent_turns),
        len(chat_continuity_text),
    )
    entities_t0 = time.perf_counter()
    resolved_entities = await assembler._ref_resolver.resolve(user_message, semantic, episodic) if user_message else []
    _log.info("[context.refs] entities_done dt=%.3fs count=%d", time.perf_counter() - entities_t0, len(resolved_entities))
    speaker_hint = data.get("speaker_hint", ("", False))
    cached_speaker_id = str(speaker_hint[0] or "").strip() if isinstance(speaker_hint, tuple) and len(speaker_hint) >= 2 and speaker_hint[1] else ""
    # 说话人识别是 recognition（认人）而非 recall（读史）。
    # Tulving (1983)：识别激活的是最近 N 条事件，不是全部历史。
    # Cowan (2001)：工作记忆有效单元约 4 个 chunk；5 条事件实用上已足够。
    speaker_context_t0 = time.perf_counter()
    interlocutor_continuity_text = await loop.run_in_executor(
        None, episodic.load_for_speaker_recognition, cached_speaker_id
    ) if cached_speaker_id else ""
    _log.info(
        "[context.refs] speaker_context_done dt=%.3fs cached_speaker=%s continuity_chars=%d",
        time.perf_counter() - speaker_context_t0,
        cached_speaker_id or "",
        len(interlocutor_continuity_text),
    )
    speaker_t0 = time.perf_counter()
    resolved_speaker = await assembler._ref_resolver.resolve_current_speaker(
        user_message,
        semantic,
        chat_id=resolved_chat_id or "",
        recent_turns=recent_turns,
        chat_continuity=chat_continuity_text,
        interlocutor_continuity=interlocutor_continuity_text,
        cached_profile_id=cached_speaker_id,
        source_hint=str(getattr(task, "source", "") or "") if task else "",
    ) if user_message else None
    _log.info(
        "[context.refs] speaker_done dt=%.3fs resolved=%s confidence=%.2f",
        time.perf_counter() - speaker_t0,
        getattr(resolved_speaker, "node_id", "") if resolved_speaker is not None else "",
        float(getattr(resolved_speaker, "confidence", 0.0) or 0.0),
    )
    if resolved_speaker is not None:
        remember_t0 = time.perf_counter()
        await assembler._ref_resolver.remember_speaker(
            resolved_speaker,
            semantic,
            task_store,
            message=user_message,
            chat_id=resolved_chat_id or "",
            task_id=task.id if task else None,
            source_hint=str(getattr(task, "source", "") or "") if task else "",
        )
        _log.info(
            "[context.refs] speaker_remembered dt=%.3fs node=%s",
            time.perf_counter() - remember_t0,
            resolved_speaker.node_id,
        )
        if resolved_speaker.node_id != cached_speaker_id or not interlocutor_continuity_text:
            refresh_t0 = time.perf_counter()
            interlocutor_continuity_text = await loop.run_in_executor(None, episodic.load_for_interlocutor_context, resolved_speaker.node_id, assembler._cfg.memory.episodic_n_recent)
            _log.info(
                "[context.refs] interlocutor_refresh_done dt=%.3fs node=%s continuity_chars=%d",
                time.perf_counter() - refresh_t0,
                resolved_speaker.node_id,
                len(interlocutor_continuity_text),
            )
    _log.info(
        "[context.refs] done dt=%.3fs entities=%d speaker=%s interlocutor_chars=%d",
        time.perf_counter() - refs_t0,
        len(resolved_entities),
        getattr(resolved_speaker, "node_id", "") if resolved_speaker is not None else "",
        len(interlocutor_continuity_text),
    )
    return {
        "resolved_entities": resolved_entities,
        "resolved_speaker": resolved_speaker,
        "entity_section": assembler._ref_resolver.format_section(resolved_entities),
        "current_interlocutor_profile_section": assembler._ref_resolver.format_speaker_section(resolved_speaker),
        "current_interlocutor_continuity_section": _fmt_interlocutor_continuity(interlocutor_continuity_text),
        "interlocutor_continuity_text": interlocutor_continuity_text,
    }


async def _assemble_context(
    assembler: Any,
    frame_or_percept: CognitionFrame | Percept,
    wm: WorkingMemory | None = None,
    task_store: TaskStore | None = None,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
    emotion: EmotionState | None = None,
    active_task: Any | None = None,
    user_message: str = "",
    chat_id: str | None = None,
    ethos_state: EthosState | None = None,
    judgment_signals: JudgmentSignals | None = None,
    hard_boundaries: list[str] | None = None,
    perception_replay: PerceptionReplaySummary | None = None,
    cognitive_signals: CognitiveSignals | None = None,
    phase: str = "initial",
    current_action: str = "",
    tool_history: list[dict[str, Any]] | None = None,
    effective_thinking: str | None = None,
    routing_overrides: dict[str, str] | None = None,
    registry_override: Any | None = None,
    runtime_life_snapshot: dict[str, Any] | None = None,
) -> str:
    percept, wm, task_store, episodic, semantic, emotion = assembler._coerce_frame_args(
        frame_or_percept, wm, task_store, episodic, semantic, emotion
    )
    wm = cast("WorkingMemory", wm)
    task_store = cast("TaskStore", task_store)
    episodic = cast("EpisodicMemory", episodic)
    semantic = cast("SemanticMemory", semantic)
    emotion = cast("EmotionState", emotion)
    task, include_open_task_overview, task_id_str, search_query, resolved_chat_id = await _resolve_context_scope(
        assembler, task_store, active_task, user_message, chat_id
    )
    loaded = await _load_context_artifacts(
        assembler,
        task_store=task_store,
        task=task,
        include_open_task_overview=include_open_task_overview,
        search_query=search_query,
        resolved_chat_id=resolved_chat_id,
        episodic=episodic,
        semantic=semantic,
    )
    refs = await _resolve_context_references(
        assembler,
        user_message=user_message,
        semantic=semantic,
        episodic=episodic,
        task_store=task_store,
        task=task,
        resolved_chat_id=resolved_chat_id,
        data=loaded["data"],
    )
    context_facts = loaded["data"]["context_facts"]
    durable_failure_snapshot = loaded["data"]["durable_failure_snapshot"]
    probes = loaded["data"].get("probes", [])
    failures = loaded["data"]["failures"]
    recent_runs = loaded["data"].get("recent_runs", [])
    waiting_tasks = loaded["data"].get("waiting_tasks", [])
    runnable_tasks = loaded["data"].get("runnable_tasks", [])
    chat_memories = loaded["data"].get("chat_memories", [])
    episodic_text = loaded["data"]["episodic_text"]
    chat_continuity_text = loaded["data"].get("chat_continuity", "")
    recent_turns = loaded["data"].get("recent_turns", [])
    similar_tasks = loaded["similar_tasks"]
    cross_task_episodic_text = loaded["cross_task_episodic_text"]
    anchors = _build_context_anchors(assembler, task, user_message, resolved_chat_id, refs["resolved_speaker"], failures)
    memories_t0 = time.perf_counter()
    _log.info(
        "[context.stage] semantic_multi_anchor_start task=%s chat=%s anchors=%d",
        task_id_str or "",
        resolved_chat_id or "",
        len(anchors),
    )
    try:
        memories = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                semantic.retrieve_multi_anchor,
                anchors,
                assembler._cfg.memory.semantic_top_k,
            ),
            timeout=8.0,
        )
    except TimeoutError:
        memories = []
        _log.warning(
            "[context.stage] semantic_multi_anchor_timeout dt=%.3fs task=%s chat=%s anchors=%d fallback=empty",
            time.perf_counter() - memories_t0,
            task_id_str or "",
            resolved_chat_id or "",
            len(anchors),
        )
    except Exception:
        memories = []
        _log.exception(
            "[context.stage] semantic_multi_anchor_failed dt=%.3fs task=%s chat=%s anchors=%d fallback=empty",
            time.perf_counter() - memories_t0,
            task_id_str or "",
            resolved_chat_id or "",
            len(anchors),
        )
    semantic_top_score = max((float(item.get("score") or 0.0) for item in memories if isinstance(item.get("score"), (int, float))), default=0.0)
    _log.info(
        "[context.stage] semantic_multi_anchor_done dt=%.3fs hits=%d top_score=%.4f",
        time.perf_counter() - memories_t0,
        len(memories),
        semantic_top_score,
    )
    should_use_daily_fallback = bool(search_query) and not cross_task_episodic_text and (not memories or semantic_top_score < assembler._cfg.memory.daily_recall_semantic_score_threshold)
    if should_use_daily_fallback:
        daily_t0 = time.perf_counter()
        daily_continuity_text = await asyncio.get_running_loop().run_in_executor(None, episodic.search_recent_daily, search_query, assembler._cfg.memory.daily_recall_days, assembler._cfg.memory.daily_recall_max_chars)
        _log.info(
            "[context.stage] daily_fallback_done dt=%.3fs chars=%d days=%d",
            time.perf_counter() - daily_t0,
            len(daily_continuity_text),
            assembler._cfg.memory.daily_recall_days,
        )
    else:
        daily_continuity_text = "（长期记忆或情节命中充分，本轮不额外注入 daily 补短）"
    recall_mode = "long_term_primary" if memories and semantic_top_score >= assembler._cfg.memory.daily_recall_semantic_score_threshold else ("episodic_cross_task" if cross_task_episodic_text else ("daily_gap_fill" if should_use_daily_fallback and daily_continuity_text and "不额外注入" not in daily_continuity_text else "no_relevant_memory"))
    breaker_facts = await task_store.list_facts(prefix="evolution:breaker:", limit=20)
    config_with_breaker = _fmt_config_snapshot(assembler._cfg) + "\n\n## 进化熔断运行时状态（runtime）\n" + _fmt_evolution_breakers(breaker_facts)
    ethos_fact = await task_store.get_fact("soul:ethos_baseline")
    soul_section = _fmt_soul(ethos_fact[0], json.dumps(assembler._cfg.soul.ethos.baseline.as_dict(), ensure_ascii=False, sort_keys=True))
    all_skills = assembler._skills.all_skills()
    skills = assembler._skills.match_for_context(
        last_applied=assembler._last_applied_skill_names,
        has_active_task=bool(task),
        has_next_step=bool(task and getattr(task, "next_step", None)),
        failure_count=len(failures) if failures else 0,
        wm_pressure=(wm.total_tokens / wm._token_budget) if wm._token_budget > 0 else 0.0,
        failure_threshold=assembler._cfg.thresholds.skill_failure_threshold,
        wm_pressure_threshold=assembler._cfg.thresholds.skill_wm_pressure_threshold,
        max_inject=assembler._cfg.thresholds.skill_max_inject,
    )
    assembler._last_selected_skills = list(skills)
    all_skills = skills + [s for s in all_skills if s.name not in {item.name for item in skills}]
    task_sections = _build_context_task_sections(
        assembler,
        task=task,
        include_open_task_overview=include_open_task_overview,
        recent_turns=recent_turns,
        recent_runs=recent_runs,
        waiting_tasks=waiting_tasks,
        runnable_tasks=runnable_tasks,
        similar_tasks=similar_tasks,
        context_facts=context_facts,
        failures=failures,
        user_message=user_message,
    )
    memory_recall_section = _fmt_memory_recall(
        query=search_query or "",
        anchors=anchors,
        chat_id=resolved_chat_id or "",
        chat_memory_hits=len(chat_memories),
        memories=memories,
        semantic_top_score=semantic_top_score,
        episodic_cross_task_hit=bool(cross_task_episodic_text),
        daily_fallback_used=bool(should_use_daily_fallback and daily_continuity_text and "不额外注入" not in daily_continuity_text),
        recall_mode=recall_mode,
    )
    memory_sections = _build_context_memory_sections(
        assembler,
        task=task,
        search_query=search_query,
        resolved_chat_id=resolved_chat_id,
        episodic_text=episodic_text,
        cross_task_episodic_text=cross_task_episodic_text,
        chat_continuity_text=chat_continuity_text,
        current_interlocutor_profile_section=refs["current_interlocutor_profile_section"],
        current_interlocutor_continuity_section=refs["current_interlocutor_continuity_section"],
        chat_memories=chat_memories,
        memories=memories,
        memory_recall_section=memory_recall_section,
        memory_system_section=_fmt_memory_system(runtime_db=str(assembler._cfg.db_path), memory_dir=str(assembler._cfg.memory_dir), workspace_dir=str(assembler._cfg.workspace_dir), semantic=semantic, memory_cfg=assembler._cfg.memory, max_concurrent_ticks=assembler._cfg.loop.max_concurrent_ticks, max_tick_queue=assembler._cfg.loop.max_tick_queue),
        daily_continuity_text=daily_continuity_text,
        entity_section=refs["entity_section"],
        soul_section=soul_section,
        skills_catalog_section=_fmt_skill_catalog(all_skills, pinned_names=set(assembler._last_applied_skill_names)),
        primary_skill_section=_fmt_primary_skill(assembler._skills.get(assembler._last_applied_skill_names[0]) if assembler._last_applied_skill_names else None),
        skills_section=_fmt_skills(skills),
        durable_failure_snapshot=durable_failure_snapshot,
        failures=failures,
    )
    state_sections = _build_context_state_sections(
        assembler,
        percept=percept,
        wm=wm,
        semantic=semantic,
        emotion=emotion,
        ethos_state=ethos_state,
        judgment_signals=judgment_signals,
        hard_boundaries=hard_boundaries,
        perception_replay=perception_replay,
        cognitive_signals=cognitive_signals,
        probes=probes,
        current_action=current_action,
        phase=phase,
        user_message=user_message,
        tool_history=tool_history,
        routing_overrides=routing_overrides,
        task=task,
        recent_turns=recent_turns,
        current_interlocutor_profile_section=refs["current_interlocutor_profile_section"],
        current_interlocutor_continuity_section=refs["current_interlocutor_continuity_section"],
        entity_section=refs["entity_section"],
        similar_tasks=similar_tasks,
        recent_runs=recent_runs,
        waiting_tasks=waiting_tasks,
        runnable_tasks=runnable_tasks,
        context_facts=context_facts,
        durable_failure_snapshot=durable_failure_snapshot,
        failures=failures,
        chat_memories=chat_memories,
        memories=memories,
        episodic_text=episodic_text,
        cross_task_episodic_text=cross_task_episodic_text,
        chat_continuity_text=chat_continuity_text,
        search_query=search_query,
        resolved_chat_id=resolved_chat_id,
        daily_continuity_text=daily_continuity_text,
        soul_section=soul_section,
        skills=skills,
        all_skills=all_skills,
        config_with_breaker=config_with_breaker,
        effective_registry=registry_override or assembler._registry,
        effective_thinking=effective_thinking,
        runtime_life_snapshot=runtime_life_snapshot,
    )
    ctx = {**task_sections, **memory_sections, **state_sections}
    return _finalize_context_text(assembler, ctx, wm)
