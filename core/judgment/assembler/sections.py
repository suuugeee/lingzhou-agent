from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.persona.self_model import fmt_self_model

from ..context.budget import apply_context_budget, resolve_judgment_prompt_budget
from ..context.sections import (
    _fmt_episodic,
    _fmt_chat_continuity,
    _fmt_chat_history,
    _fmt_chat_memories,
    _fmt_cross_task_episodic,
    _fmt_current_time,
    _fmt_ethos,
    _fmt_life_state,
    _fmt_memories,
    _fmt_percept,
    _fmt_shell_capabilities,
    _fmt_tools,
    _fmt_wm,
)
from ..context.signals import (
    _fmt_hard_boundaries,
    _fmt_judgment_signals,
    _fmt_perception_replay,
    _fmt_risk_sections,
    _fmt_uncertainty_sections,
    _fmt_wm_proposal_sections,
)
from ..context.skills import (
    _fmt_blind_spots,
    _fmt_cognitive_signals,
    _fmt_probe_sensors,
)
from ..context.tasks import (
    _fmt_context_facts,
    _fmt_durable_failures,
    _fmt_failures,
    _fmt_recent_runs,
    _fmt_runnable_tasks,
    _fmt_similar_tasks,
    _fmt_task,
    _fmt_waiting_tasks,
)
from ..context.utils import _fill_template, _validate_context_schema
from ..output import _build_team_view_from_cfg

if TYPE_CHECKING:
    from core.perception import (
        CognitiveSignals,
        EmotionState,
        EthosState,
        JudgmentSignals,
        Percept,
        PerceptionReplaySummary,
    )
    from core.skill import Skill
    from memory.working import WorkingMemory
    from store.semantic import SemanticMemory


_log = logging.getLogger("lingzhou.judgment")


def _build_context_task_sections(
    assembler: Any,
    *,
    task: Any | None,
    include_open_task_overview: bool,
    recent_turns: list[Any],
    recent_runs: list[Any],
    waiting_tasks: list[Any],
    runnable_tasks: list[Any],
    similar_tasks: list[Any],
    context_facts: list[Any],
) -> dict[str, Any]:
    task_id = task.id if task else None
    sections = {
        "task_section": _fmt_task(task),
        "task_facts_section": _fmt_context_facts(context_facts),
        "recent_runs_section": _fmt_recent_runs(recent_runs),
        "chat_history_section": _fmt_chat_history(recent_turns, max_chars=assembler._cfg.thresholds.chat_history_max_chars),
    }
    if include_open_task_overview:
        sections.update({
            "waiting_tasks_section": _fmt_waiting_tasks(waiting_tasks),
            "runnable_tasks_section": _fmt_runnable_tasks(runnable_tasks, active_task_id=task_id),
            "similar_tasks_section": _fmt_similar_tasks(similar_tasks),
        })
    else:
        sections.update({"waiting_tasks_section": "", "runnable_tasks_section": "", "similar_tasks_section": ""})
    return sections


def _build_context_memory_sections(
    assembler: Any,
    *,
    task: Any | None,
    search_query: str,
    resolved_chat_id: str,
    episodic_text: str,
    cross_task_episodic_text: str,
    chat_continuity_text: str,
    current_interlocutor_profile_section: str,
    current_interlocutor_continuity_section: str,
    chat_memories: list[Any],
    memories: list[Any],
    memory_recall_section: str,
    memory_system_section: str,
    daily_continuity_text: str,
    entity_section: str,
    soul_section: str,
    skills_catalog_section: str,
    primary_skill_section: str,
    skills_section: str,
    durable_failure_snapshot: Any,
    failures: list[Any],
) -> dict[str, Any]:
    return {
        "episodic_section": _fmt_episodic(episodic_text),
        "cross_task_episodic_section": _fmt_cross_task_episodic(cross_task_episodic_text),
        "chat_continuity_section": _fmt_chat_continuity(chat_continuity_text),
        "current_interlocutor_profile_section": current_interlocutor_profile_section,
        "current_interlocutor_continuity_section": current_interlocutor_continuity_section,
        "daily_continuity_section": (_fmt_chat_continuity(daily_continuity_text) if daily_continuity_text else "（近两日无相关 daily 补短）"),
        "entity_section": entity_section,
        "chat_memory_section": _fmt_chat_memories(chat_memories),
        "memories_section": _fmt_memories(memories),
        "memory_recall_section": memory_recall_section,
        "memory_system_section": memory_system_section,
        "soul_section": soul_section,
        "skills_catalog_section": skills_catalog_section,
        "primary_skill_section": primary_skill_section,
        "skills_section": skills_section,
        "failures_section": _fmt_failures(failures),
        "durable_failure_section": _fmt_durable_failures(durable_failure_snapshot),
    }


def _build_context_state_sections(
    assembler: Any,
    *,
    percept: Percept,
    wm: WorkingMemory,
    semantic: SemanticMemory,
    emotion: EmotionState,
    ethos_state: EthosState | None,
    judgment_signals: JudgmentSignals | None,
    hard_boundaries: list[str] | None,
    perception_replay: PerceptionReplaySummary | None,
    cognitive_signals: CognitiveSignals | None,
    probes: list[Any],
    current_action: str,
    phase: str,
    user_message: str,
    tool_history: list[dict[str, Any]] | None,
    routing_overrides: dict[str, str] | None,
    task: Any | None,
    recent_turns: list[Any],
    current_interlocutor_profile_section: str,
    current_interlocutor_continuity_section: str,
    entity_section: str,
    similar_tasks: list[Any],
    recent_runs: list[Any],
    waiting_tasks: list[Any],
    runnable_tasks: list[Any],
    context_facts: list[Any],
    durable_failure_snapshot: Any,
    failures: list[Any],
    chat_memories: list[Any],
    memories: list[Any],
    episodic_text: str,
    cross_task_episodic_text: str,
    chat_continuity_text: str,
    search_query: str,
    resolved_chat_id: str,
    daily_continuity_text: str,
    soul_section: str,
    skills: list[Skill],
    all_skills: list[Skill],
    config_with_breaker: str,
    effective_registry: Any,
    runtime_life_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    _wm_items = wm.get_top(15)
    state_sections = {
        "emotion_valence": f"{emotion.valence:.2f}",
        "emotion_arousal": f"{emotion.arousal:.2f}",
        "emotion_dominant": emotion.dominant or "（未确定）",
        "emotion_regulation": f"{emotion.regulation.strategy}（{emotion.regulation.reason}）" if emotion.regulation.reason else emotion.regulation.strategy,
        "wm_section": _fmt_wm(_wm_items, wm_count=len(wm), wm_capacity=wm._capacity, wm_tokens=wm.total_tokens, wm_token_budget=wm._token_budget),
        "wm_proposal_sections": _fmt_wm_proposal_sections(_wm_items),
        "tools_section": _fmt_tools(effective_registry.list_manifests()),
        "shell_capabilities_section": _fmt_shell_capabilities(),
        "perception_section": _fmt_percept(percept),
        "ethos_section": _fmt_ethos(ethos_state),
        "signals_section": _fmt_judgment_signals(judgment_signals),
        "hard_boundaries_section": _fmt_hard_boundaries(hard_boundaries),
        "perception_replay_section": _fmt_perception_replay(perception_replay),
        "cognitive_signals_section": _fmt_cognitive_signals(cognitive_signals),
        "risk_sections": _fmt_risk_sections(
            judgment_signals=judgment_signals,
            failures=failures,
            durable_failure_snapshot=durable_failure_snapshot,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
        ),
        "uncertainty_sections": _fmt_uncertainty_sections(
            judgment_signals=judgment_signals,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
        ),
        "probe_sensors_section": _fmt_probe_sensors(probes),
        "blind_spot_section": _fmt_blind_spots(probes),
        "self_model_section": fmt_self_model(assembler._executor.self_model),
        "life_state_section": _fmt_life_state(runtime_life_snapshot),
        "team_view": _build_team_view_from_cfg(assembler._cfg),
        "model_routing_section": assembler._build_model_routing_section(
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            effective_thinking=assembler._cfg.thinking,
            routing_overrides=routing_overrides,
            registry=effective_registry,
        ),
        "current_time_section": _fmt_current_time(),
        "config_section": config_with_breaker,
        "user_message": user_message or "",
    }
    return state_sections


def _finalize_context_text(assembler: Any, ctx: dict[str, Any], wm: WorkingMemory) -> str:
    _validate_context_schema(ctx)
    catalog_path = assembler._cfg.workspace_dir / "models.json"
    budgets = []
    for tier in ("reader", "reasoner", "repair"):
        _, model_ref = assembler._executor._resolve_tier_model(tier)
        budgets.append(resolve_judgment_prompt_budget(assembler._cfg, model_ref, catalog_path=catalog_path))
    budget = min(budgets) if budgets else assembler._cfg.judgment_input_token_budget()
    if budget > 0 and wm is not None and wm._token_budget > 0:
        wm._token_budget = max(256, int(budget * assembler._cfg.memory.wm_token_budget_ratio))
    ctx = apply_context_budget(ctx, budget, skill_min_tokens=assembler._cfg.thresholds.skill_min_budget_tokens)
    assembler._last_context_sections = dict(ctx)
    assembler._last_context_budget = int(budget)
    if budget:
        used = sum(len(v) for v in ctx.values())
        assembler._executor.self_model.context_budget = f"{budget // 1000}K" if budget >= 1000 else str(budget)
        assembler._executor.self_model.context_pressure = min(1.0, used / max(budget, 1))
    return _fill_template(assembler._judgment_template, ctx)
