"""运行时构造器。

这里集中创建 lingzhou 的器官；CognitionLoop 只保留 façade 与旧调用点兼容。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.evolution import EvolutionEngine
from core.execution import ExecutionLayer
from core.judgment import JudgmentLayer
from core.loop.drive.behavior import BehaviorTracker
from core.metabolic import MetabolicEngine
from core.perception import EmotionState
from core.persona import IdentityBootstrapManager
from core.probe import ProbeManager
from core.resource_guard import local_embedding_memory_preflight
from memory.working import WorkingMemory
from provider import create_provider, create_provider_with_model
from provider.base import EmbeddingProvider
from store.episodic import EpisodicMemory
from store.semantic import SemanticMemory
from store.task import TaskStore
from tools.registry import ToolRegistry

from ..cycle.dispatcher import ConcurrentTickDispatcher
from ..runs.driver import RunDriver
from .context import RuntimeContext

if TYPE_CHECKING:
    from core.config import Config

_log = logging.getLogger("lingzhou.loop")


@dataclass
class EmbeddingRuntime:
    embed_fn: Callable[..., Any] | None
    provider: Any | None = None


def _elapsed(started: float) -> float:
    return time.monotonic() - started


def build_runtime_context(cfg: Config, owner: Any) -> RuntimeContext:
    """构造完整运行时器官上下文。"""
    init_started = time.monotonic()

    stage_started = time.monotonic()
    registry = ToolRegistry()
    repo_root = Path(__file__).resolve().parents[3]
    tools_dir = repo_root / "tools"
    registry.discover(tools_dir)
    _log.info("[startup] loop tools discovered dt=%.3fs", _elapsed(stage_started))

    stage_started = time.monotonic()
    from core.plugin import PluginManager

    plugins_dir = repo_root / "plugins"
    plugin_manager = PluginManager(plugins_dir)
    plugin_manager.discover()
    plugin_manager.load_all()
    plugin_manager.register_all(tool_registry=registry)
    plugin_manager.start_all()
    _log.info("[plugin] 已加载 %d 个插件", len(plugin_manager.list_plugins()))
    _log.info("[startup] loop plugins ready dt=%.3fs", _elapsed(stage_started))

    stage_started = time.monotonic()
    wm = WorkingMemory(
        capacity=cfg.memory.working_capacity,
        token_budget=cfg.effective_wm_token_budget(),
        item_max_tokens=cfg.memory.wm_item_max_tokens,
    )
    episodic = EpisodicMemory(cfg.memory_dir, max_events=cfg.memory.max_events)
    task_store = TaskStore(Path(cfg.db_path))
    _log.info("[startup] loop base memory ready dt=%.3fs", _elapsed(stage_started))

    emotion = EmotionState.from_config(cfg)

    stage_started = time.monotonic()
    provider = create_provider(cfg)
    from core.perception import PerceptionLayer

    perception = PerceptionLayer(cfg)
    judgment = JudgmentLayer(provider, registry, cfg)
    execution = ExecutionLayer(registry, cfg)
    run_driver = RunDriver(execution)
    evolution = EvolutionEngine(cfg, provider, registry)
    _log.info("[startup] loop cognition layers ready dt=%.3fs", _elapsed(stage_started))

    embedding_runtime = _build_embedding_runtime(cfg)
    stage_started = time.monotonic()
    _log.info("[startup] semantic init start")
    semantic = SemanticMemory(
        cfg.memory_dir,
        decay_lambda=cfg.memory.semantic_decay_lambda,
        embed_fn=embedding_runtime.embed_fn,
        embedding_weight=cfg.memory.embedding_weight,
        source_weight=cfg.memory.semantic_source_weight,
        temporal_weight=cfg.memory.semantic_temporal_weight,
        temporal_window_days=cfg.memory.semantic_temporal_window_days,
    )
    _log.info("[startup] semantic init done dt=%.3fs", _elapsed(stage_started))
    metabolic = MetabolicEngine(task_store, semantic_memory=semantic)

    stage_started = time.monotonic()
    soul = IdentityBootstrapManager(cfg, task_store, wm)
    behavior = BehaviorTracker(
        wait_streak_notify=list(cfg.loop.wait_streak_notify),
        streak_threshold=cfg.loop.behavior_streak_threshold,
        wm_priorities={
            "behavior_loop": cfg.thresholds.wm_pri_user_msg,
            "edit_caution": cfg.thresholds.wm_pri_self_aware,
            "belief_stale": cfg.thresholds.wm_pri_critical,
        },
        registry=registry,
        seq_window_warn_at=cfg.thresholds.behavior_seq_window_warn_at,
        seq_window_gap_ratio=cfg.thresholds.behavior_seq_window_gap_ratio,
        belief_stale_threshold=cfg.thresholds.behavior_belief_stale_threshold,
        belief_window=cfg.thresholds.behavior_belief_window,
    )

    from core.loop.drive.engine import SelfDriveEngine

    self_drive = SelfDriveEngine(str(cfg.db_path))

    probe_file = cfg.workspace_dir / "probes.json"
    probe_manager = ProbeManager(probe_file)
    judgment._assembler._probe_manager = probe_manager

    cfg_file = cfg._base_dir / "lingzhou.json"
    from store.auth import AUTH_PROFILES_PATH

    tick_dispatcher = ConcurrentTickDispatcher(
        owner,
        max_concurrent=cfg.loop.max_concurrent_ticks,
        max_queue=cfg.loop.max_tick_queue,
    )

    context = RuntimeContext(
        _cfg=cfg,
        _registry=registry,
        _plugin_manager=plugin_manager,
        _wm=wm,
        _episodic=episodic,
        _task_store=task_store,
        _emotion=emotion,
        _provider=provider,
        _embedding_provider=embedding_runtime.provider,
        _perception=perception,
        _judgment=judgment,
        _execution=execution,
        _run_driver=run_driver,
        _evolution=evolution,
        _routing_providers={},
        _semantic=semantic,
        _metabolic=metabolic,
        _soul=soul,
        _behavior=behavior,
        _self_drive=self_drive,
        _probe_manager=probe_manager,
        _cfg_file=cfg_file,
        _cfg_mtime=cfg_file.stat().st_mtime if cfg_file.exists() else 0.0,
        _auth_profiles_path=AUTH_PROFILES_PATH,
        _auth_profiles_mtime=AUTH_PROFILES_PATH.stat().st_mtime if AUTH_PROFILES_PATH.exists() else 0.0,
        _tick_dispatcher=tick_dispatcher,
    )
    _log.info("[startup] loop runtime objects ready dt=%.3fs", _elapsed(stage_started))
    _log.info("[startup] loop construct complete dt=%.3fs", _elapsed(init_started))
    return context


def _build_embedding_runtime(cfg: Config) -> EmbeddingRuntime:
    mode = str(getattr(cfg.memory, "embedding_provider", "local") or "local").strip()
    mode_lower = mode.lower()
    if mode_lower == "none":
        _log.info("[embedding] disabled by memory.embedding_provider=none")
        return EmbeddingRuntime(None)

    if mode_lower == "local":
        return EmbeddingRuntime(_build_local_embedding_fn(cfg))

    return _build_remote_embedding_runtime(cfg, mode)


def _build_local_embedding_fn(cfg: Config) -> Callable[..., Any] | None:
    if cfg.memory.local_embed_model:
        preflight = local_embedding_memory_preflight(
            model=cfg.memory.local_embed_model,
            min_available_mib=cfg.memory.local_embed_min_available_mib,
            guard_enabled=cfg.memory.local_embed_command_guard,
        )
        if not preflight.ok:
            _log.warning(
                "[loop] 本地 embedding 模型加载被资源守卫跳过: model=%s available_mib=%s required_mib=%s reason=%s",
                cfg.memory.local_embed_model,
                preflight.available_mib,
                preflight.required_mib,
                preflight.reason,
            )
            return None
        try:
            import importlib
            import os as _os

            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            st_kwargs: dict[str, Any] = {}
            if cfg.memory.local_embed_cache_dir:
                st_kwargs["cache_folder"] = cfg.memory.local_embed_cache_dir
            st_module = importlib.import_module("sentence_transformers")
            sentence_transformer = st_module.SentenceTransformer
            local_st = sentence_transformer(cfg.memory.local_embed_model, **st_kwargs)

            def _local_embed(texts: list[str]) -> list[list[float]]:
                return local_st.encode(texts, normalize_embeddings=True).tolist()

            return _local_embed
        except Exception as exc:
            _log.warning("[loop] 本地 embedding 模型加载失败，回退到 FTS: %s", exc)
            return None
    _log.info("[embedding] local provider enabled but local_embed_model is empty, fallback=%s", cfg.memory.embedding_fallback)
    return None


def _provider_embed_fn(cfg: Config, provider: Any | None) -> Callable[..., Any] | None:
    if cfg.memory.embedding_model and _provider_supports_embedding(provider):
        provider_ref = str(getattr(provider, "model_ref", provider.__class__.__name__))
        _log.info("[embedding] provider enabled provider=%s model=%s", provider_ref, cfg.memory.embedding_model)
        return provider.embed
    if cfg.memory.embedding_model:
        provider_ref = str(getattr(provider, "model_ref", provider.__class__.__name__ if provider is not None else "none"))
        _log.warning("[embedding] provider unavailable provider=%s model=%s fallback=%s", provider_ref, cfg.memory.embedding_model, cfg.memory.embedding_fallback)
    return None


def _build_remote_embedding_runtime(cfg: Config, provider_name: str) -> EmbeddingRuntime:
    if not cfg.memory.embedding_model:
        _log.warning("[embedding] memory.embedding_provider=%s ignored: embedding_model is empty", provider_name)
        return EmbeddingRuntime(None)
    if provider_name not in cfg.providers:
        _log.warning("[embedding] memory.embedding_provider=%s not found, fallback=%s", provider_name, cfg.memory.embedding_fallback)
        return EmbeddingRuntime(None)
    if cfg.providers[provider_name].mode == "codex":
        _log.warning("[embedding] provider=%s mode=codex 不支持 embeddings, fallback=%s", provider_name, cfg.memory.embedding_fallback)
        return EmbeddingRuntime(None)
    try:
        provider = create_provider_with_model(cfg, f"{provider_name}/{cfg.memory.embedding_model}")
    except Exception as exc:
        _log.warning("[embedding] provider=%s 创建失败, fallback=%s: %s", provider_name, cfg.memory.embedding_fallback, exc)
        return EmbeddingRuntime(None)
    if not _provider_supports_embedding(provider):
        _log.warning("[embedding] provider=%s 不支持 embeddings, fallback=%s", provider_name, cfg.memory.embedding_fallback)
        return EmbeddingRuntime(None, provider=provider)
    _log.info("[embedding] independent provider enabled provider=%s model=%s", provider_name, cfg.memory.embedding_model)
    return EmbeddingRuntime(provider.embed, provider=provider)


def _provider_supports_embedding(provider: Any | None) -> bool:
    if provider is None:
        return False
    if str(getattr(provider, "_provider_mode", "") or "").strip() == "codex":
        return False
    return isinstance(provider, EmbeddingProvider)
