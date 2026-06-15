from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.config import Config
from core.evolution import EvolutionEngine
from core.execution import ExecutionLayer
from core.judgment import JudgmentLayer
from core.log_fields import format_log_fields
from core.loop.routing_overrides import normalize_routing_overrides
from core.perception import PerceptionLayer
from provider import create_provider

from ..runs.driver import RunDriver
from .builder import EmbeddingRuntime, _build_embedding_runtime
from .startup import _build_routing_providers

if TYPE_CHECKING:
    from pathlib import Path

    from .main import CognitionLoop

_log = logging.getLogger("lingzhou.loop")


@dataclass
class _HotReloadCandidate:
    cfg: Config
    provider: Any
    embedding_runtime: EmbeddingRuntime
    routing_providers: dict[str, Any]
    judgment: JudgmentLayer
    execution: ExecutionLayer
    evolution: EvolutionEngine
    perception: PerceptionLayer
    replaced_provider_stack: bool

    async def close_new_stack(self) -> None:
        await _close_provider_stack(
            self.provider if self.replaced_provider_stack else None,
            self.routing_providers if self.replaced_provider_stack else {},
            self.embedding_runtime.provider,
        )


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _provider_reload_signature(cfg: Config) -> str:
    payload = {
        "model": cfg.model,
        "routing": cfg.routing,
        "providers": {
            name: provider.model_dump(mode="json")
            for name, provider in sorted(cfg.providers.items())
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def _close_provider_stack(
    provider: Any | None,
    routing_providers: dict[str, Any],
    embedding_provider: Any | None = None,
) -> None:
    closed: set[int] = set()
    for target in (embedding_provider, provider):
        if target is None or id(target) in closed:
            continue
        closed.add(id(target))
        with contextlib.suppress(Exception):
            await target.close()
    for routing_provider in routing_providers.values():
        if id(routing_provider) in closed:
            continue
        closed.add(id(routing_provider))
        with contextlib.suppress(Exception):
            await routing_provider.close()


async def _refresh_runtime_routing_overrides(loop: CognitionLoop) -> None:
    task_store = getattr(loop, "_task_store", None)
    if task_store is None:
        return

    refreshed: dict[str, str] | None = None
    try:
        raw, found = await task_store.get_fact("pref:routing_overrides")
    except Exception as exc:
        _log.warning("[hot-reload] 读取 DB routing_overrides 失败,保留当前内存态: %s", exc)
        return

    if found and raw:
        try:
            payload = json.loads(raw)
            refreshed = normalize_routing_overrides(payload)
        except Exception as exc:
            _log.warning("[hot-reload] DB routing_overrides 解析失败,已清空内存态: %s", exc)

    previous = getattr(loop, "_pending_routing_overrides", None)
    loop._pending_routing_overrides = refreshed
    if previous != refreshed:
        _log.info(
            "[hot-reload] routing_overrides %s",
            format_log_fields(overrides=refreshed or {}),
        )


async def _build_reload_candidate(
    loop: CognitionLoop,
    new_cfg: Config,
    *,
    auth_changed: bool,
) -> _HotReloadCandidate:
    replaced_provider_stack = auth_changed or _provider_reload_signature(loop._cfg) != _provider_reload_signature(new_cfg)
    provider = loop._provider
    routing_providers = loop._routing_providers
    new_provider: Any | None = None
    new_routing_providers: dict[str, Any] | None = None
    embedding_runtime: EmbeddingRuntime | None = None
    try:
        if replaced_provider_stack:
            new_provider = create_provider(new_cfg)
            new_routing_providers = _build_routing_providers(new_cfg)
            provider = new_provider
            routing_providers = new_routing_providers
        embedding_runtime = _build_embedding_runtime(new_cfg)
        judgment = JudgmentLayer(provider, loop._registry, new_cfg)
        judgment.self_model = loop._judgment.self_model
        judgment.self_model.set_routing(new_cfg)
        judgment.set_routing_providers(routing_providers)

        return _HotReloadCandidate(
            cfg=new_cfg,
            provider=provider,
            embedding_runtime=embedding_runtime,
            routing_providers=routing_providers,
            judgment=judgment,
            execution=ExecutionLayer(loop._registry, new_cfg),
            evolution=EvolutionEngine(new_cfg, provider, loop._registry),
            perception=PerceptionLayer(new_cfg),
            replaced_provider_stack=replaced_provider_stack,
        )
    except Exception:
        await _close_provider_stack(
            new_provider,
            new_routing_providers or {},
            embedding_runtime.provider if embedding_runtime is not None else None,
        )
        raise


async def _commit_hot_reload_candidate(
    loop: CognitionLoop,
    candidate: _HotReloadCandidate,
    *,
    cfg_mtime: float,
    auth_mtime: float,
) -> None:
    old_provider = loop._provider
    old_embedding_provider = getattr(loop, "_embedding_provider", None)
    old_routing_providers = loop._routing_providers

    loop._cfg = candidate.cfg
    loop._provider = candidate.provider
    loop._embedding_provider = candidate.embedding_runtime.provider
    loop._routing_providers = candidate.routing_providers
    loop._judgment = candidate.judgment
    loop._execution = candidate.execution
    loop._run_driver = RunDriver(loop._execution)  # Phase 3b: rewire
    loop._evolution = candidate.evolution
    loop._perception = candidate.perception
    loop._cfg_mtime = cfg_mtime
    loop._auth_profiles_mtime = auth_mtime
    loop._soul._cfg = candidate.cfg
    loop._semantic._embed_fn = candidate.embedding_runtime.embed_fn
    loop._semantic._embedding_weight = loop._cfg.memory.embedding_weight
    await _refresh_runtime_routing_overrides(loop)

    try:
        await loop._soul.refresh_identity(loop._judgment)
    except Exception as exc:
        _log.warning("[hot-reload] 身份前缀刷新失败,保留新运行时: %s", exc)

    if candidate.replaced_provider_stack:
        await _close_provider_stack(old_provider, old_routing_providers, old_embedding_provider)
    elif old_embedding_provider is not None and old_embedding_provider is not loop._embedding_provider:
        await _close_provider_stack(provider=None, routing_providers={}, embedding_provider=old_embedding_provider)


async def _maybe_hot_reload_provider_impl(loop: CognitionLoop) -> None:
    cfg_file = getattr(loop, "_cfg_file", None)
    if cfg_file is None:
        return
    if not cfg_file.exists():
        return

    cfg_mtime = _file_mtime(cfg_file)
    auth_mtime = _file_mtime(loop._auth_profiles_path)
    cfg_changed = cfg_mtime > loop._cfg_mtime
    auth_changed = auth_mtime > loop._auth_profiles_mtime
    if not cfg_changed and not auth_changed:
        return

    try:
        new_cfg = Config.load(cfg_file)
    except Exception as exc:
        _log.warning("[hot-reload] 配置解析失败,保留旧运行时: %s", exc)
        return

    old_model = loop._cfg.model
    try:
        candidate = await _build_reload_candidate(loop, new_cfg, auth_changed=auth_changed)
    except Exception as exc:
        _log.warning("[hot-reload] 构建候选运行时失败,保留旧运行时: %s", exc)
        return

    try:
        await _commit_hot_reload_candidate(loop, candidate, cfg_mtime=cfg_mtime, auth_mtime=auth_mtime)
    except Exception as exc:
        await candidate.close_new_stack()
        _log.warning("[hot-reload] 提交新运行时失败,已回退旧运行时: %s", exc)
        return

    if auth_changed and not cfg_changed:
        _log.info("[hot-reload] auth profiles 更新,已原子重建 provider 栈")
    elif candidate.replaced_provider_stack and old_model != new_cfg.model:
        _log.info(
            "[hot-reload] model_swap %s",
            format_log_fields(old_model=old_model, model_ref=new_cfg.model),
        )
    else:
        _log.info("[hot-reload] 配置热加载完成")
