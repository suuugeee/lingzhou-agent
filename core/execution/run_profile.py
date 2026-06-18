"""core/execution/run_profile.py — 执行层 run_type/worker 映射策略。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from core.judgment.tiers import READER_TIER, REASONER_TIER
from core.execution.task_tier import TASK_DEFAULT_TIER, resolve_task_model_tier as _resolve_task_model_tier
from tools.registry import tool_has_capability

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

_log = logging.getLogger("lingzhou.execution")


@dataclass(frozen=True)
class ExecutionProfile:
    run_type: str
    worker_type: str
    model_tier: str


@dataclass(frozen=True)
class RunProfile:
    run_type: str
    worker_type: str


@dataclass(frozen=True)
class RunTypeProfile:
    """run_type 的统一元数据定义。"""

    run_type: str
    worker_type: str
    default_tier: str


RUN_TYPE_EVOLUTION = "evolve"
RUN_TYPE_EXEC = "exec"
RUN_TYPE_LLM = "llm"
RUN_TYPE_MULTIMODAL = "multimodal"
RUN_TYPE_CHAT_REPLY = "chat_reply"
RUN_TYPE_SUBAGENT = "subagent"
RUN_TYPE_JUDGE = "judge"
RUN_TYPE_PROBE = "probe"
RUN_TYPE_TOOL_CHAIN = "tool_chain"

WORKER_EVOLVE = "evolve-worker"
WORKER_EXEC = "exec-worker"
WORKER_LLM = "llm-worker"
WORKER_MULTIMODAL = "multimodal-worker"
WORKER_SUBAGENT = "subagent-worker"
WORKER_TOOL_CHAIN = "tool-chain-worker"
WORKER_LIMIT_DEFAULT_KEY = "max_tool_chain_workers"
WORKER_LIMITS_CONFIG_KEY = {
    WORKER_TOOL_CHAIN: WORKER_LIMIT_DEFAULT_KEY,
    WORKER_EVOLVE: WORKER_LIMIT_DEFAULT_KEY,
    WORKER_EXEC: "max_exec_workers",
    WORKER_MULTIMODAL: "max_multimodal_workers",
    WORKER_LLM: "max_llm_workers",
}

RUN_TYPE_PROFILES: tuple[RunTypeProfile, ...] = (
    RunTypeProfile(RUN_TYPE_JUDGE, WORKER_TOOL_CHAIN, READER_TIER),
    RunTypeProfile(RUN_TYPE_TOOL_CHAIN, WORKER_TOOL_CHAIN, TASK_DEFAULT_TIER),
    RunTypeProfile(RUN_TYPE_CHAT_REPLY, WORKER_TOOL_CHAIN, READER_TIER),
    RunTypeProfile(RUN_TYPE_EVOLUTION, WORKER_EVOLVE, REASONER_TIER),
    RunTypeProfile(RUN_TYPE_SUBAGENT, WORKER_SUBAGENT, TASK_DEFAULT_TIER),
    RunTypeProfile(RUN_TYPE_LLM, WORKER_LLM, READER_TIER),
    RunTypeProfile(RUN_TYPE_EXEC, WORKER_EXEC, TASK_DEFAULT_TIER),
    RunTypeProfile(RUN_TYPE_MULTIMODAL, WORKER_MULTIMODAL, TASK_DEFAULT_TIER),
    RunTypeProfile(RUN_TYPE_PROBE, WORKER_TOOL_CHAIN, READER_TIER),
)

KNOWN_RUN_TYPES: tuple[str, ...] = tuple(profile.run_type for profile in RUN_TYPE_PROFILES)
RUN_TYPE_PROFILE_MAP: dict[str, RunTypeProfile] = {profile.run_type: profile for profile in RUN_TYPE_PROFILES}
RUN_TYPE_DEFAULT_TIER: dict[str, str] = {run_type: profile.default_tier for run_type, profile in RUN_TYPE_PROFILE_MAP.items()}
_DEFAULT_RUN_TYPE_PROFILE = RUN_TYPE_PROFILE_MAP[RUN_TYPE_TOOL_CHAIN]


def _normalize_run_type(run_type: str) -> str:
    """标准化 run_type 文本，供映射查找复用。"""
    return str(run_type or "").strip().lower()


def run_type_profile(run_type: str) -> RunTypeProfile:
    """返回 run_type 的规范化 profile；未知类型回退为 tool-chain profile。"""
    return RUN_TYPE_PROFILE_MAP.get(_normalize_run_type(run_type), _DEFAULT_RUN_TYPE_PROFILE)


def resolve_default_tier_for_run_type(run_type: str, run_type_routing: dict[str, str] | None) -> str:
    """在给定 routing 下返回 run_type 的默认档位。

    规则：
    1. 如果 routing 有匹配，优先使用。
    2. 否则回退 profile 默认值（未知 run_type 则回退 tool-chain 默认）。
    """
    normalized = _normalize_run_type(run_type)
    if normalized:
        mapped = run_type_routing.get(normalized) if isinstance(run_type_routing, dict) else None
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
    profile = run_type_profile(normalized)
    return profile.default_tier


def worker_limit_config_key(worker_type: str) -> str:
    """返回 worker_type 对应的运行时并发配置 key，不存在时回退到 tool chain 配置。"""
    return WORKER_LIMITS_CONFIG_KEY.get(worker_type, WORKER_LIMIT_DEFAULT_KEY)


def worker_type_for_run_type(run_type: str) -> str:
    """返回 run_type 对应的 worker_type；无匹配时回退到工具链 worker。"""
    return run_type_profile(run_type).worker_type

DEFAULT_RUN_PROFILE = RunProfile(
    run_type=RUN_TYPE_TOOL_CHAIN,
    worker_type=WORKER_TOOL_CHAIN,
)

_EVOLUTION_TOOL_PROFILE = {
    "evolution.evolve": RunProfile(RUN_TYPE_EVOLUTION, WORKER_EVOLVE),
    "evolution.synthesize": RunProfile(RUN_TYPE_EVOLUTION, WORKER_EVOLVE),
}
_TOOL_NAME_PROFILE_MAP = {
    "subagent.run": RunProfile(RUN_TYPE_SUBAGENT, WORKER_SUBAGENT),
    **_EVOLUTION_TOOL_PROFILE,
}
_MONITOR_SIGNAL_KEYS: tuple[str, ...] = ("monitor_fact_key", "status_fact_key")
_CAPABILITY_PROFILE_SEQUENCE: tuple[tuple[str, RunProfile], ...] = (
    ("run_spawn", RunProfile(RUN_TYPE_EXEC, WORKER_EXEC)),
    ("multimodal", RunProfile(RUN_TYPE_MULTIMODAL, WORKER_MULTIMODAL)),
)


def _coerce_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return {}


def _has_monitor_signal(params: dict[str, Any]) -> bool:
    return any(bool(params.get(key)) for key in _MONITOR_SIGNAL_KEYS)


def _infer_by_name(tool_name: str) -> RunProfile | None:
    return _TOOL_NAME_PROFILE_MAP.get(tool_name)


def _infer_by_capability(
    tool_name: str,
    registry: ToolRegistry | None,
) -> RunProfile | None:
    if registry is None:
        return None
    for cap, profile in _CAPABILITY_PROFILE_SEQUENCE:
        if tool_has_capability(registry, tool_name, cap):
            return profile
    return None


def infer_run_profile(
    tool_name: str,
    params: dict[str, Any] | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> tuple[str, str]:
    """按工具名和参数推导 execution 层运行档位（run_type）与 worker 名。"""
    params_map = _coerce_params(params)

    explicit_profile = _infer_by_name(tool_name)
    if explicit_profile is not None:
        return explicit_profile.run_type, explicit_profile.worker_type

    if _has_monitor_signal(params_map):
        _log.debug("[run-profile] tool=%s classified as llm-worker via fact monitor", tool_name)
        return RUN_TYPE_LLM, WORKER_LLM

    capability_profile = _infer_by_capability(tool_name, registry)
    if capability_profile is not None:
        return capability_profile.run_type, capability_profile.worker_type

    return DEFAULT_RUN_PROFILE.run_type, DEFAULT_RUN_PROFILE.worker_type


def resolve_execution_profile(
    task_tier: str | None,
    tool_name: str,
    *,
    run_type_routing: dict[str, str] | None,
    params: dict[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[str, str, str]:
    """返回执行分发的三元组：run_type / worker_type / model_tier。"""
    profile = resolve_execution_dispatch(
        task_tier=task_tier,
        tool_name=tool_name,
        run_type_routing=run_type_routing,
        params=params,
        registry=registry,
    )
    return profile.run_type, profile.worker_type, profile.model_tier


def resolve_execution_dispatch(
    task_tier: str | None,
    tool_name: str,
    *,
    run_type_routing: dict[str, str] | None,
    params: dict[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> ExecutionProfile:
    """返回执行分发结构化结果：run_type / worker_type / model_tier。"""
    run_type, worker_type = infer_run_profile(
        tool_name,
        params=params,
        registry=registry,
    )
    model_tier = _resolve_task_model_tier(
        task_tier=task_tier or "",
        run_type=run_type,
        run_type_routing=run_type_routing or {},
        resolve_default_tier_for_run_type=resolve_default_tier_for_run_type,
    )
    return ExecutionProfile(
        run_type=run_type,
        worker_type=worker_type,
        model_tier=model_tier,
    )


def iter_run_profiles() -> Iterable[RunProfile]:
    """返回当前支持的显式 run profile 集，用于配置联动/调试。"""
    explicit_profiles = {profile for profile in _TOOL_NAME_PROFILE_MAP.values()}
    capability_profiles = {profile for _, profile in _CAPABILITY_PROFILE_SEQUENCE}
    return sorted(
        {*explicit_profiles, *capability_profiles, DEFAULT_RUN_PROFILE},
        key=lambda p: (p.run_type, p.worker_type),
    )
