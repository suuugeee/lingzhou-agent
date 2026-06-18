"""core.loop.runtime.startup — loop 启动装配与状态恢复。"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.loop.routing_overrides import normalize_routing_overrides_payload
from core.metabolic import add_run, set_soul_fact
from core.execution.run_profile import RUN_TYPE_JUDGE, run_type_profile
from core.persona.self_model import SelfModel
from provider import create_provider_with_model
from provider.models_gen import ensure_models_json

if TYPE_CHECKING:
    from core.config import Config

_log = logging.getLogger("lingzhou.loop")

# ── 运行时启动自检（与 cli/diag doctor 共享字段清单）────────────────────────────

_MEMORY_SCHEMA_FIELDS: tuple[str, ...] = ("wm_item_max_tokens",)

_THRESHOLDS_SCHEMA_FIELDS: tuple[str, ...] = (
    "skill_max_inject",
    "skill_failure_threshold",
    "skill_wm_pressure_threshold",
    "skill_min_budget_tokens",
)

# 兼容 diag 旧名（仅计数用）
_MEMORY_FIELD_PATCHES = dict.fromkeys(_MEMORY_SCHEMA_FIELDS)
_THRESHOLDS_FIELD_PATCHES = dict.fromkeys(_THRESHOLDS_SCHEMA_FIELDS)


def _missing_config_schema_fields() -> dict[str, list[str]]:
    """检查 config_models 是否包含运行所需字段；返回 {类名: 缺失字段}。"""
    from core.config_models import MemoryConfig, ThresholdsConfig

    specs: list[tuple[str, type, tuple[str, ...]]] = [
        ("MemoryConfig", MemoryConfig, _MEMORY_SCHEMA_FIELDS),
        ("ThresholdsConfig", ThresholdsConfig, _THRESHOLDS_SCHEMA_FIELDS),
    ]
    missing_by_class: dict[str, list[str]] = {}
    for class_name, cls, required in specs:
        try:
            instance = cls()
        except Exception:
            missing_by_class[class_name] = list(required)
            continue
        missing = [field for field in required if not hasattr(instance, field)]
        if missing:
            missing_by_class[class_name] = missing
    return missing_by_class


def _startup_health_check(cfg: Config, project_root: Path) -> None:
    """运行时启动自检（非阻塞，仅 warn 级日志）：

    1. 确保 memory_dir / workspace_dir / db_parent 目录存在
    2. Config schema 字段校验（MemoryConfig + ThresholdsConfig）
    3. 宪法器官校验（A3）：加载 CONSTITUTION.md 并缓存哈希
    """
    # 1. 目录自动创建
    for label, raw_path in [
        ("memory_dir",    cfg.memory_dir),
        ("workspace_dir", cfg.workspace_dir),
        ("db_parent",     Path(cfg.db_path).parent),
    ]:
        p = Path(raw_path).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _log.warning("[startup] %s 无法创建，可能导致运行异常: %s  路径=%s", label, exc, p)

    # 2. Config schema 校验
    _ = project_root
    for class_name, fields in _missing_config_schema_fields().items():
        _log.warning(
            "[startup] %s schema 不完整，缺少字段: %s（请升级代码库）",
            class_name,
            fields,
        )

    # 3. 宪法器官校验（公理 A3 / A4）
    from core.immune.constitution import load_constitution
    constitution_text = load_constitution(cfg.constitution_path)
    if not constitution_text:
        _log.warning(
            "[immune] 宪法文件缺失或为空，免疫器官将在无宪法状态下运行。"
            "请完成 workspace 初始化（`lingzhou init`）以生成 CONSTITUTION.md。"
            "路径: %s",
            cfg.constitution_path,
        )


def _build_routing_providers(cfg: Config) -> dict[str, Any]:
    """根据 cfg.routing 构建分层路由 providers 字典。"""
    if not cfg.routing:
        return {}
    from provider.catalog import lookup_model, lookup_model_ref
    catalog_path = cfg.workspace_dir / "models.json"
    providers: dict[str, Any] = {}
    for tier, model_ref in cfg.routing.items():
        if not model_ref or model_ref == cfg.model:
            continue
        # 启动期校验：若 model_id 不在指定 provider 的目录中，但存在于其他 provider，提前告警
        if "/" in model_ref and lookup_model_ref(model_ref, catalog_path=catalog_path) is None:
            model_id = model_ref.split("/", 1)[1]
            provider_name = model_ref.split("/", 1)[0]
            alt = lookup_model(model_id, catalog_path=catalog_path)
            if alt is not None:
                _log.warning(
                    "[routing] tier=%s model=%s: 模型 %r 未在 provider %r 的内置目录中注册，"
                    "但在其他 provider 中存在。请检查 routing 配置是否正确（provider 名写错）。",
                    tier, model_ref, model_id, provider_name,
                )
        try:
            providers[tier] = create_provider_with_model(cfg, model_ref)
            _log.info("[routing] tier=%s model=%s", tier, model_ref)
        except Exception as exc:
            _log.warning("[routing] tier=%s model=%s 创建失败,跳过: %s", tier, model_ref, exc)
    return providers


def _routing_summary_text(cfg: Config, routing_providers: dict[str, Any]) -> str:
    routing_lines: list[str] = []
    for tier, model_ref in cfg.routing.items():
        if model_ref == cfg.model:
            routing_lines.append(f"  {tier}: {model_ref} (= main, no separate provider)")
        elif tier in routing_providers:
            routing_lines.append(f"  {tier}: {model_ref} ✓")
        else:
            routing_lines.append(f"  {tier}: {model_ref} ✗ MISSING - provider 创建失败,实际回退至 {cfg.model}")
    if cfg.routing and not routing_providers:
        _log.warning(
            "[routing] 所有 routing provider 均创建失败,整个 routing 降级为单模型 %s。"
            "请检查各 provider 的 API key 环境变量是否已设置。",
            cfg.model,
        )
    return "\n".join(routing_lines) if routing_lines else "  (无路由配置,全部使用主模型)"


def _runtime_config_snapshot(
    cfg: Config,
    routing_providers: dict[str, Any],
    *,
    stage: str,
) -> tuple[str, str]:
    config_path = (cfg._base_dir / "lingzhou.json").resolve()
    routing_items = ", ".join(
        f"{tier}={model_ref}" for tier, model_ref in sorted(cfg.routing.items())
    ) if cfg.routing else "(none)"
    startup_line = (
        "[startup] "
        f"stage={stage} config={config_path} "
        f"main_model={cfg.model} routing={routing_items}"
    )
    return startup_line, _routing_summary_text(cfg, routing_providers)


def _log_runtime_config(
    cfg: Config,
    routing_providers: dict[str, Any],
    *,
    stage: str,
) -> str:
    startup_line, routing_summary = _runtime_config_snapshot(
        cfg,
        routing_providers,
        stage=stage,
    )
    _log.info(startup_line)
    _log.info("[routing] effective summary:\n%s", routing_summary)
    return routing_summary


async def _open_runtime_impl(loop: Any) -> None:
    from core.paths import project_root as _project_root
    _startup_health_check(loop._cfg, _project_root())
    await loop._task_store.open()
    await ensure_models_json(loop._cfg)
    loop._routing_providers = _build_routing_providers(loop._cfg)
    _log_runtime_config(loop._cfg, loop._routing_providers, stage="open")
    loop._judgment.set_routing_providers(loop._routing_providers)
    loop._bootstrap_mode = await loop._soul.bootstrap(loop._judgment, run_kind="interactive")
    loop._judgment.self_model.record_start(name="lingzhou")
    loop._judgment.self_model.set_routing(loop._cfg)
    await _restore_self_model_impl(loop)
    # 探针系统：从 probes.json 加载（已在 ProbeManager.__init__ 同步完成），启动调度 Task
    await loop._probe_manager.start(loop._wm, loop_ref=loop)
    await _restore_state_from_db_impl(loop)


async def _prepare_runtime_run_impl(loop: Any) -> tuple[Config, str]:
    from core.paths import project_root as _project_root
    started = time.monotonic()
    stage_started = time.monotonic()
    _startup_health_check(loop._cfg, _project_root())
    _log.info("[startup] phase=health_check done dt=%.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    await loop._task_store.open()
    _log.info("[startup] phase=task_store.open done dt=%.3fs", time.monotonic() - stage_started)
    cfg = loop._cfg
    stage_started = time.monotonic()
    await ensure_models_json(cfg)
    _log.info("[startup] phase=ensure_models_json done dt=%.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    loop._routing_providers = _build_routing_providers(cfg)
    _log.info("[startup] phase=routing_providers done dt=%.3fs", time.monotonic() - stage_started)
    routing_summary = _log_runtime_config(cfg, loop._routing_providers, stage="run")
    loop._judgment.set_routing_providers(loop._routing_providers)
    stage_started = time.monotonic()
    loop._bootstrap_mode = await loop._soul.bootstrap(loop._judgment, run_kind="interactive")
    _log.info("[startup] phase=soul.bootstrap done dt=%.3fs", time.monotonic() - stage_started)
    loop._judgment.self_model.record_start(name="lingzhou")
    loop._judgment.self_model.set_routing(cfg)
    stage_started = time.monotonic()
    await _restore_self_model_impl(loop)
    _log.info("[startup] phase=restore_self_model done dt=%.3fs", time.monotonic() - stage_started)
    # 探针系统：从 probes.json 加载（已在 ProbeManager.__init__ 同步完成），启动调度 Task
    stage_started = time.monotonic()
    await loop._probe_manager.start(loop._wm, loop_ref=loop)
    _log.info("[startup] phase=probe.start done dt=%.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    await _restore_state_from_db_impl(loop)
    _log.info("[startup] phase=restore_state done dt=%.3fs", time.monotonic() - stage_started)
    _log.info("[startup] runtime prepare done dt=%.3fs", time.monotonic() - started)
    return cfg, routing_summary


async def _restore_state_from_db_impl(loop: Any) -> None:
    """从 DB 恢复上次持久化的状态，实现跨重启连续性。"""
    # ── soul:born_at：首次启动写入，后续恢复；记录灵舟的诞生时刻 ──
    import time as _time
    born_json, born_found = await loop._task_store.get_fact("soul:born_at")
    if born_found and born_json:
        with contextlib.suppress(Exception):
            loop._judgment.self_model.born_at = float(born_json)
    else:
        _born_ts = _time.time()
        await set_soul_fact(
            loop,
            key="soul:born_at",
            value=str(_born_ts),
            scope="system",
            source="loop/startup/born_at",
            decision_basis="bootstrap initialization born_at marker",
        )
        loop._judgment.self_model.born_at = _born_ts
        _log.info("[startup] soul:born_at 首次写入: %.0f", _born_ts)

    emotion_json, emotion_found = await loop._task_store.get_fact("soul:emotion_state")
    if emotion_found and emotion_json:
        try:
            emotion = json.loads(emotion_json)
            loop._emotion.valence = float(emotion.get("valence", loop._emotion.valence))
            loop._emotion.arousal = float(emotion.get("arousal", loop._emotion.arousal))
            loop._emotion.dominance = float(emotion.get("dominance", loop._emotion.dominance))
        except Exception:
            pass

    overrides_json, overrides_found = await loop._task_store.get_fact("pref:routing_overrides")
    if overrides_found and overrides_json:
        try:
            loop._pending_routing_overrides = normalize_routing_overrides_payload(
                overrides_json,
                catalog_path=loop._cfg.workspace_dir / "models.json",
            )
            if loop._pending_routing_overrides:
                _log.info("[routing] 从 DB 恢复 routing_overrides: %s", loop._pending_routing_overrides)
        except Exception:
            pass

    zombie_count = await loop._task_store.reset_in_progress_tasks()
    if zombie_count > 0:
        _log.info("[restart] 重置 %d 个 in_progress 任务为 pending", zombie_count)

    # Phase 3d：清理上次崩溃遗留的非终态 Run
    try:
        stale_count = await loop._task_store.cancel_stale_runs()
        if stale_count > 0:
            _log.info("[restart] 取消 %d 个遗留 stale Run（崩溃/重启恢复）", stale_count)
    except Exception as _exc:
        _log.debug("[restart] cancel_stale_runs 失败（不影响启动）: %s", _exc)

    # Phase 3d：若无 pending Run，写入 bootstrap pending Run（确保 poll 可以驱动首轮 tick）
    try:
        _existing_pending = await loop._task_store.get_pending_runs(limit=1)
        if not _existing_pending:
            judge_profile = run_type_profile(RUN_TYPE_JUDGE)
            _bootstrap_run_id = await add_run(
                loop,
                run_type=judge_profile.run_type,
                worker_type=judge_profile.worker_type,
                status="pending",
                log_text="[startup] bootstrap pending Run — awaiting first poll",
                source="loop/runtime/startup/bootstrap",
            )
            _log.info("[startup] 创建 bootstrap pending Run #%d", _bootstrap_run_id)
    except Exception as _exc:
        _log.debug("[startup] bootstrap pending Run 创建失败（不影响启动）: %s", _exc)

    # ── 崩溃连续性恢复：读取 survival.json，若上次非干净退出则注入 WM 感知 ──
    _inject_crash_recovery(loop)


def _inject_crash_recovery(loop: Any) -> None:
    """若上次为崩溃退出，将崩溃摘要注入 WM，让 LLM 在第一轮 tick 感知到。"""
    try:
        _sp = loop._cfg.state_dir / "survival.json"
        if not _sp.exists():
            return
        snap = json.loads(_sp.read_text(encoding="utf-8"))
        if snap.get("exit_type") == "clean":
            return
        # 上次是 crash 退出
        tick = snap.get("tick", "?")
        ts = snap.get("ts", "未知时间")
        task_title = snap.get("active_task_title") or "无"
        task_goal = snap.get("active_task_goal") or ""
        last_action = snap.get("last_action") or "未知"
        emotion = snap.get("emotion") or {}
        valence = emotion.get("valence", getattr(loop._emotion, "valence", 0))
        arousal = emotion.get("arousal", getattr(loop._emotion, "arousal", 0))

        content = (
            f"[崩溃恢复] 上次运行在 tick={tick}（{ts}）异常终止（非干净退出）。\n"
            f"  中断前活跃任务: 「{task_title}」\n"
        )
        if task_goal:
            content += f"  任务目标: {task_goal}\n"
        content += (
            f"  最后动作: {last_action}\n"
            f"  情绪状态: valence={valence} arousal={arousal}\n"
            "  建议: 先确认中断前的任务是否需要继续，检查是否有遗留副作用，"
            "再决定本轮行动。"
        )
        from memory.working import WMItem
        loop._wm.add(WMItem(
            kind="crash_recovery",
            content=content,
            priority=0.97,
        ))
        _log.info("[startup] 注入崩溃恢复信号: tick=%s ts=%s", tick, ts)
    except Exception as exc:
        _log.debug("[startup] 读取 survival.json 失败: %s", exc)


async def _restore_self_model_impl(loop: Any) -> None:
    """从 DB 恢复自我模型(跨重启连续性)。"""
    raw, found = await loop._task_store.get_fact("self:model")
    if found and raw:
        loop._judgment.self_model = SelfModel.from_json(raw, name="lingzhou")
        loop._judgment.self_model.set_routing(loop._cfg)
        loop._judgment.self_model.tick_count = 0
        _log.info(
            "[self_model] 已恢复: api=%d tokens=%d (tick=0 重置)",
            loop._judgment.self_model.api_call_count,
            loop._judgment.self_model.total_tokens,
        )
