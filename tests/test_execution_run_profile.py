from __future__ import annotations

from typing import Any

from core.execution.run_profile import (
    WORKER_LIMIT_DEFAULT_KEY,
    RUN_TYPE_EVOLUTION,
    RUN_TYPE_EXEC,
    RUN_TYPE_PROFILES,
    RUN_TYPE_LLM,
    RUN_TYPE_MULTIMODAL,
    RUN_TYPE_SUBAGENT,
    RUN_TYPE_TOOL_CHAIN,
    RUN_TYPE_JUDGE,
    TASK_DEFAULT_TIER,
    WORKER_EVOLVE,
    WORKER_EXEC,
    WORKER_LLM,
    WORKER_MULTIMODAL,
    WORKER_SUBAGENT,
    WORKER_TOOL_CHAIN,
    run_type_profile,
    resolve_default_tier_for_run_type,
    ExecutionProfile,
    resolve_execution_profile,
    infer_run_profile,
    resolve_execution_dispatch,
    worker_limit_config_key,
    worker_type_for_run_type,
)
from tools.registry import ToolEntry, ToolManifest


class _Registry:
    def __init__(self) -> None:
        self._tools = {
            "demo.exec": ToolEntry(
                manifest=ToolManifest(name="demo.exec", description="demo", capabilities=("run_spawn",)),
                handler=lambda params, ctx: None,  # type: ignore[return-value]
            ),
            "demo.vision": ToolEntry(
                manifest=ToolManifest(name="demo.vision", description="demo", capabilities=("multimodal",)),
                handler=lambda params, ctx: None,  # type: ignore[return-value]
            ),
        }

    def get(self, name: str) -> Any:
        return self._tools.get(name)


def test_infer_run_profile_defaults_to_tool_chain():
    assert infer_run_profile("unknown.tool") == (RUN_TYPE_TOOL_CHAIN, WORKER_TOOL_CHAIN)


def test_infer_run_profile_explicit_subagent_name_has_priority():
    assert infer_run_profile("subagent.run") == (RUN_TYPE_SUBAGENT, WORKER_SUBAGENT)


def test_infer_run_profile_monitor_signal_forces_llm():
    assert infer_run_profile(
        "demo.exec",
        params={"monitor_fact_key": "run:1"},
        registry=_Registry(),
    ) == (RUN_TYPE_LLM, WORKER_LLM)


def test_infer_run_profile_capability_dispatches_exec_and_multimodal():
    registry = _Registry()
    assert infer_run_profile("demo.exec", registry=registry) == (RUN_TYPE_EXEC, WORKER_EXEC)
    assert infer_run_profile("demo.vision", registry=registry) == (RUN_TYPE_MULTIMODAL, WORKER_MULTIMODAL)


def test_infer_run_profile_evolution_tool_name():
    assert infer_run_profile("evolution.evolve") == (RUN_TYPE_EVOLUTION, WORKER_EVOLVE)
    assert infer_run_profile("evolution.synthesize") == (RUN_TYPE_EVOLUTION, WORKER_EVOLVE)


def test_worker_limit_config_key_maps_known_workers_and_defaults():
    assert worker_limit_config_key(WORKER_EXEC) == "max_exec_workers"
    assert worker_limit_config_key(WORKER_MULTIMODAL) == "max_multimodal_workers"
    assert worker_limit_config_key(WORKER_LLM) == "max_llm_workers"
    assert worker_limit_config_key(WORKER_EVOLVE) == WORKER_LIMIT_DEFAULT_KEY
    assert worker_limit_config_key("unknown-worker") == WORKER_LIMIT_DEFAULT_KEY


def test_worker_type_for_run_type_uses_execution_mapping_with_fallback():
    assert worker_type_for_run_type(RUN_TYPE_EXEC) == WORKER_EXEC
    assert worker_type_for_run_type(RUN_TYPE_JUDGE) == WORKER_TOOL_CHAIN


def test_resolve_execution_profile_preserves_explicit_task_tier():
    assert resolve_execution_profile(
        "reader",
        "demo.exec",
        run_type_routing={RUN_TYPE_EXEC: "reasoner"},
    ) == (RUN_TYPE_EXEC, WORKER_EXEC, "reader")


def test_resolve_execution_profile_switches_by_monitor_signal():
    assert resolve_execution_profile(
        TASK_DEFAULT_TIER,
        "demo.exec",
        run_type_routing={RUN_TYPE_LLM: "repair", RUN_TYPE_EXEC: "reasoner"},
        params={"monitor_fact_key": "run:1"},
        registry=_Registry(),
    ) == (RUN_TYPE_LLM, WORKER_LLM, "repair")


def test_resolve_execution_dispatch_returns_structured_profile():
    assert resolve_execution_dispatch(
        TASK_DEFAULT_TIER,
        "demo.exec",
        run_type_routing={TASK_DEFAULT_TIER: "reader", RUN_TYPE_EXEC: "reasoner"},
        registry=_Registry(),
    ) == ExecutionProfile(
        run_type=RUN_TYPE_EXEC,
        worker_type=WORKER_EXEC,
        model_tier="reasoner",
    )


def test_resolve_execution_dispatch_preserves_unknown_toolchain_defaults():
    assert resolve_execution_dispatch(
        TASK_DEFAULT_TIER,
        "unknown.tool",
        run_type_routing={},
        registry=_Registry(),
    ) == ExecutionProfile(
        run_type=RUN_TYPE_TOOL_CHAIN,
        worker_type=WORKER_TOOL_CHAIN,
        model_tier=TASK_DEFAULT_TIER,
    )


def test_run_type_profile_map_is_single_source_for_default_tier_and_worker():
    for profile in RUN_TYPE_PROFILES:
        assert profile.default_tier == run_type_profile(profile.run_type).default_tier
        assert profile.worker_type == run_type_profile(profile.run_type).worker_type


def test_run_type_profile_falls_back_to_tool_chain_for_unknown():
    fallback = run_type_profile("unknown.run_type")
    assert fallback.run_type == RUN_TYPE_TOOL_CHAIN
    assert fallback.worker_type == WORKER_TOOL_CHAIN
    assert fallback.default_tier == TASK_DEFAULT_TIER


def test_resolve_default_tier_for_run_type_prefers_routing_and_preserves_profile_default():
    routing = {RUN_TYPE_JUDGE: "reasoner", "custom_type": "repair"}
    assert resolve_default_tier_for_run_type(RUN_TYPE_JUDGE, routing) == "reasoner"
    assert resolve_default_tier_for_run_type("custom_type", routing) == "repair"
    assert resolve_default_tier_for_run_type("missing_type", routing) == TASK_DEFAULT_TIER
    assert resolve_default_tier_for_run_type(" ", routing) == TASK_DEFAULT_TIER


def test_resolve_default_tier_for_run_type_is_case_insensitive_and_trims():
    routing = {RUN_TYPE_JUDGE: "reader"}
    assert resolve_default_tier_for_run_type("  JUDGE ", routing) == "reader"


def test_task_tier_resolution_defaults_to_task_default_when_no_task_tier_and_route_is_task_default():
    from core.execution.task_tier import resolve_task_model_tier

    assert resolve_task_model_tier(
        task_tier="",
        run_type=RUN_TYPE_EXEC,
        run_type_routing={RUN_TYPE_EXEC: TASK_DEFAULT_TIER},
        resolve_default_tier_for_run_type=resolve_default_tier_for_run_type,
    ) == TASK_DEFAULT_TIER


def test_task_tier_resolution_normalizes_and_rewrites_invalid_task_tier_hint():
    from core.execution.task_tier import resolve_task_model_tier
    from core.judgment.tiers import READER_TIER

    assert resolve_task_model_tier(
        task_tier=" Reader ",
        run_type=RUN_TYPE_EXEC,
        run_type_routing={RUN_TYPE_EXEC: "reasoner"},
        resolve_default_tier_for_run_type=resolve_default_tier_for_run_type,
    ) == READER_TIER

    assert resolve_task_model_tier(
        task_tier="INVALID",
        run_type=RUN_TYPE_EXEC,
        run_type_routing={RUN_TYPE_EXEC: "reasoner"},
        resolve_default_tier_for_run_type=resolve_default_tier_for_run_type,
    ) == "reasoner"
