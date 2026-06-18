from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any

from core.judgment.policy.routing_context import (
    analyze_tool_history_budget,
    continue_phase_policy_payload,
)
from provider.catalog import lookup_model
from core.judgment.tiers import (
    JUDGMENT_TIERS,
    tier_display_label,
    tier_routing_description,
)

from ..output import registry_manifest_signature, tool_tier_mapping

_TIER_DESCRIPTIONS: dict[str, str] = {
    tier: tier_routing_description(tier) or "" for tier in JUDGMENT_TIERS
}

_ROUTING_TIERS = JUDGMENT_TIERS
_NEXT_PHASE_TIER_GUIDE = "• next_phase_tier：" + "，".join(
    f"{tier}={tier_display_label(tier)}（{tier_routing_description(tier) or tier}）" for tier in JUDGMENT_TIERS
)


def _fmt_duration_ms(value: float) -> str:
    ms = float(value)
    return f"{ms / 1000.0:g}s" if ms >= 1000 else f"{ms:g}ms"


@functools.lru_cache(maxsize=4)
def _catalog_models_snapshot(catalog_key: str) -> tuple[dict[str, Any], ...]:
    """models.json 目录快照（按路径缓存，避免每 tick 全量扫描）。"""
    from provider import catalog as _cat

    path = Path(catalog_key) if catalog_key else None
    return tuple(
        {
            "model": f"{provider_name}/{model.get('id', '')}",
            "provider": provider_name,
            "reasoning": bool(model.get("reasoning")),
            "context_window": model.get("context_window"),
        }
        for provider_name in _cat.list_providers(catalog_path=path)
        for model in _cat.list_provider_models(provider_name, catalog_path=path)
    )


@functools.lru_cache(maxsize=16)
def _capability_mapping_snapshot(
    signature: tuple[tuple[str, str | None, str, tuple[str, ...]], ...],
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for name, _prefer_tier, _progress_category, caps in signature:
        for cap in caps:
            mapping.setdefault(cap, []).append(name)
    return {cap: tuple(sorted(names)) for cap, names in mapping.items()}


def _model_id_from_ref(model_ref: str) -> str:
    return model_ref.split("/", 1)[1] if "/" in model_ref else model_ref


def _lookup_workspace_model_spec(cfg: Any, model_ref: str) -> dict[str, Any]:
    spec = lookup_model(_model_id_from_ref(model_ref), catalog_path=cfg.workspace_dir / "models.json")
    return spec or {}


def _current_action_capabilities(registry: Any, current_action: str) -> list[str]:
    for manifest in registry.list_manifests():
        if manifest.name == current_action:
            return sorted(manifest.capabilities)
    return []


def _build_available_models(
    assembler: Any,
    *,
    routing_overrides: dict[str, str] | None,
    effective_thinking: str,
    cfg: Any,
) -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    now = time.time()
    for tier in _ROUTING_TIERS:
        _, model_ref = assembler._executor._resolve_tier_model(tier)
        key = (tier, model_ref)
        if key in seen:
            continue
        seen.add(key)
        spec = _lookup_workspace_model_spec(cfg, model_ref)
        reasoning = bool(spec.get("reasoning"))
        health = assembler._executor._get_health(model_ref)
        override_model = (routing_overrides or {}).get(tier)
        available.append({
            "tier": tier,
            "model": model_ref,
            "available": assembler._executor._is_model_available(model_ref),
            "reasoning": reasoning,
            "cost_level": assembler._executor._cost_level_for_model(model_ref, reasoning),
            "latency_level": assembler._executor._latency_level_for_model(model_ref, reasoning),
            "context_window": spec.get("context_window") or cfg.context_window_tokens,
            "current_thinking": effective_thinking or cfg.thinking,
            "last_error": assembler._executor._provider_errors.get(model_ref),
            "last_error_code": health.last_code or None,
            "cooldown_remaining_sec": max(0, int(health.cooldown_until - now)),
            "overridden_by": override_model if override_model and override_model != model_ref else None,
        })
    return available


def _delegation_guide_text(cfg: Any) -> str:
    with_task_bounds = cfg.loop.idle_with_task_bounds
    no_task_bounds = cfg.loop.idle_no_task_bounds
    with_task_bounds_text = f"{_fmt_duration_ms(with_task_bounds[0])}-{_fmt_duration_ms(with_task_bounds[1])}"
    no_task_bounds_text = f"{_fmt_duration_ms(no_task_bounds[0])}-{_fmt_duration_ms(no_task_bounds[1])}"
    default_gap_text = (
        f"有任务 {_fmt_duration_ms(cfg.loop.active_idle_gap)}，"
        f"无任务 {_fmt_duration_ms(cfg.loop.max_idle_gap)}"
    )
    return (
        "你是当前层的决策者，可以通过 model_strategy 中的字段调控下一轮行为。"
        f"{_NEXT_PHASE_TIER_GUIDE}"
        "• tool_tier_mapping：runtime 当前对工具族的默认分层真相。"
        "• tool_capability_mapping：runtime 注入的工具能力真相，优先按能力标签推理。"
        "• continue_phase_policy：若 tool_history_will_compact_next=true，下一轮早期工具记录会折叠。"
        f"• idle 参考：当前有任务时 {with_task_bounds_text}，无任务时 {no_task_bounds_text}。"
        f"• 当前 loop 默认备用值（{default_gap_text}）。"
        "• next_idle_gap_secs / next_idle_gap_ms：必须设置其一，ms 优先。"
        "• routing_overrides：临时覆盖 tier→model 映射。"
        "• thinking_override：覆盖下一轮的 thinking 等级。"
    )


def _build_model_routing_section(
    assembler: Any,
    *,
    phase: str,
    user_message: str,
    current_action: str,
    tool_history: list[dict[str, Any]] | None,
    effective_thinking: str,
    routing_overrides: dict[str, str] | None = None,
    registry: Any | None = None,
) -> str:
    effective_registry = registry or assembler._registry
    cfg = assembler._cfg

    available_models = _build_available_models(
        assembler,
        routing_overrides=routing_overrides,
        effective_thinking=effective_thinking,
        cfg=cfg,
    )

    budget = analyze_tool_history_budget(
        effective_registry,
        cfg,
        tool_history,
        user_message=user_message,
    )

    manifest_signature = registry_manifest_signature(effective_registry)
    capability_mapping = _capability_mapping_snapshot(manifest_signature)
    current_action_caps = _current_action_capabilities(effective_registry, current_action)

    tool_history_count = len(tool_history or [])

    payload: dict[str, Any] = {
        "active_overrides": routing_overrides or {},
        "available_models": available_models,
        "tool_tier_mapping": tool_tier_mapping(effective_registry),
        "tool_capability_mapping": {k: list(v) for k, v in capability_mapping.items()},
        "current_action_capabilities": current_action_caps,
        "continue_phase_policy": continue_phase_policy_payload(cfg, tool_history_count),
        "tier_descriptions": dict(_TIER_DESCRIPTIONS),
        "delegation_guide": _delegation_guide_text(cfg),
        "budget_state": {
            "task_explore_count": budget.task_explore_count,
            "repeat_action_count": budget.repeat_action_count,
            "repeat_read_count": budget.repeat_read_count,
            "ask_evidence_hits": budget.ask_evidence_hits,
            "ask_evidence_budget": cfg.thresholds.ask_evidence_budget,
            "task_explore_converge_after": cfg.thresholds.task_explore_converge_after,
            "global_cost_posture": budget.global_cost_posture,
        },
        "routing_hint": {
            "phase": phase,
            "current_action": current_action,
            "user_message_present": bool(user_message),
        },
    }
    catalog_key = str((cfg.workspace_dir / "models.json").resolve())
    catalog_models = list(_catalog_models_snapshot(catalog_key))
    compact_model_routing = bool(
        getattr(getattr(cfg, "thresholds", None), "compact_model_routing", True)
    )
    payload["catalog_models"] = (
        [
            {
                "model": item.get("model"),
                "reasoning": item.get("reasoning"),
                "context_window": item.get("context_window"),
            }
            for item in catalog_models
        ]
        if compact_model_routing
        else catalog_models
    )
    payload["primary_provider"] = {"model": cfg.model}
    if hasattr(assembler, "_ref_resolver") and assembler._ref_resolver is not None:
        resolver = assembler._ref_resolver
        payload["reference_resolution"] = {
            "llm_available": resolver.llm_available,
            "last_error": resolver.last_llm_error,
            "last_error_code": resolver.last_llm_error_code,
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)
