from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

from cli.common import console

_RUNTIME_ROUTING_TIERS = frozenset({"reader", "reasoner", "repair"})
_MODEL_TARGET_ALIASES = {
    "": "primary",
    "model": "primary",
    "main": "primary",
    "primary": "primary",
    "vision": "vision",
    "image": "vision",
    "image.analyze": "vision",
    "vision_model": "vision",
    "识图": "vision",
    "视图": "vision",
    "reasoner": "reasoner",
    "reader": "reader",
    "repair": "repair",
}


def _provider_name(model_ref: str) -> str:
    provider, _, _ = model_ref.partition("/")
    return provider


def _validate_model_ref(model_ref: str) -> None:
    """Reject common accidental model refs before they are persisted."""
    provider, sep, model_id = str(model_ref or "").partition("/")
    if not provider or not sep or not model_id:
        raise ValueError(f"模型引用必须是 provider/model-id 格式: {model_ref!r}")
    if model_id.strip().isdigit():
        try:
            from provider.catalog import list_provider_models

            catalog_ids = {
                str(model.get("id") or "")
                for model in list_provider_models(provider)
                if isinstance(model, dict)
            }
        except Exception:
            catalog_ids = set()
        if model_id not in catalog_ids:
            raise ValueError(
                f"模型 ID 不能是编号 {model_id!r}；请使用完整模型名，例如 {provider}/gpt-5.5"
            )


def _normalize_model_target(target: str) -> str:
    return _MODEL_TARGET_ALIASES.get((target or "").strip().lower(), (target or "").strip())


def _effective_target_model(cfg_data: dict[str, Any], target: str) -> str:
    normalized = _normalize_model_target(target)
    if normalized == "primary":
        return str(cfg_data.get("model") or "")
    if normalized == "vision":
        return str(cfg_data.get("vision_model") or cfg_data.get("model") or "")
    routing = cfg_data.get("routing")
    if isinstance(routing, dict):
        model_ref = routing.get(normalized)
        if isinstance(model_ref, str) and model_ref:
            return model_ref
    return str(cfg_data.get("model") or "")


def _apply_model_target_selection(
    cfg_data: dict[str, Any],
    *,
    current_model: str,
    new_model: str,
    target: str,
) -> dict[str, Any]:
    normalized = _normalize_model_target(target)
    _validate_model_ref(new_model)
    if normalized == "primary":
        previous = str(cfg_data.get("model") or current_model)
        cfg_data["model"] = new_model
        return {
            "target": "primary",
            "previous": previous,
            "routing_changed": _sync_routing_models_on_primary_switch(
                cfg_data,
                old_model=current_model,
                new_model=new_model,
            ),
            "runtime_override_tier": None,
        }
    if normalized == "vision":
        previous = _effective_target_model(cfg_data, normalized)
        cfg_data["vision_model"] = new_model
        return {
            "target": "vision",
            "previous": previous,
            "routing_changed": ["vision_model"],
            "runtime_override_tier": None,
        }
    if normalized not in _RUNTIME_ROUTING_TIERS:
        raise ValueError(f"未知模型目标: {target!r}；可用值: primary, vision, reader, reasoner, repair")

    routing = cfg_data.get("routing")
    if not isinstance(routing, dict):
        routing = {}
        cfg_data["routing"] = routing

    previous = _effective_target_model(cfg_data, normalized)
    routing[normalized] = new_model
    return {
        "target": normalized,
        "previous": previous,
        "routing_changed": [normalized],
        "runtime_override_tier": normalized if normalized in _RUNTIME_ROUTING_TIERS else None,
    }


def _merge_runtime_routing_override(overrides: dict[str, str], *, tier: str, model_ref: str) -> dict[str, str]:
    merged = {
        key: value
        for key, value in overrides.items()
        if key in _RUNTIME_ROUTING_TIERS and isinstance(value, str) and value
    }
    merged[tier] = model_ref
    return merged


def _set_db_routing_override(cfg_path: Path, *, tier: str, model_ref: str) -> None:
    if tier not in _RUNTIME_ROUTING_TIERS or not model_ref:
        return

    import sqlite3 as _sqlite3

    try:
        from core.config import Config as _Config

        cfg = _Config.load(cfg_path)
        db_path = cfg.db_path
        if not db_path.exists():
            return

        conn = _sqlite3.connect(str(db_path), timeout=30)
        try:
            row = conn.execute(
                "SELECT value FROM facts WHERE key='pref:routing_overrides'"
            ).fetchone()
            overrides: dict[str, str] = {}
            if row and row[0]:
                payload = _json.loads(row[0])
                if isinstance(payload, dict):
                    overrides = {
                        key: value
                        for key, value in payload.items()
                        if isinstance(key, str) and isinstance(value, str)
                    }

            merged = _merge_runtime_routing_override(overrides, tier=tier, model_ref=model_ref)
            serialized = _json.dumps(merged, ensure_ascii=False)
            if row:
                conn.execute(
                    "UPDATE facts SET value=?, scope='system', updated_at=datetime('now') WHERE key='pref:routing_overrides'",
                    (serialized,),
                )
            else:
                conn.execute(
                    "INSERT INTO facts (key, value, scope, updated_at) VALUES ('pref:routing_overrides', ?, 'system', datetime('now'))",
                    (serialized,),
                )
            conn.commit()
        finally:
            conn.close()

        console.print(
            f"[green]✓ 运行时 {tier} override 已同步:[/green] [bold cyan]{model_ref}[/bold cyan]"
        )
    except Exception as exc:
        console.print(f"[yellow]⚠ 运行时 {tier} override 同步失败（非致命）: {exc}[/yellow]")


def _sync_routing_models_on_primary_switch(
    cfg_data: dict,
    *,
    old_model: str,
    new_model: str,
) -> list[str]:
    """切换主模型时，仅同步跟随主模型的 routing 条目。

    规则：
    - 常规切换时，仅同步精确指向旧主模型的 routing 条目。
    - 若用户重选当前主模型，则修复仍停留在同 provider 旧模型上的残留 routing 条目。

    这样既能让主推理链路跟随当前主模型，又不会覆盖 reader 等明确分流到其他 provider 的配置。
    """
    if not old_model or not new_model:
        return []
    routing = cfg_data.get("routing")
    if not isinstance(routing, dict):
        return []

    repair_same_provider = old_model == new_model
    new_provider = _provider_name(new_model)
    changed: list[str] = []
    for tier, model_ref in routing.items():
        if tier not in _RUNTIME_ROUTING_TIERS:
            continue
        if not isinstance(model_ref, str) or model_ref == new_model:
            continue
        if model_ref == old_model or (
            repair_same_provider
            and new_provider
            and _provider_name(model_ref) == new_provider
        ):
            routing[tier] = new_model
            changed.append(str(tier))
    return changed


def _sync_db_routing_overrides(cfg_path: Path, *, old_model: str, new_model: str) -> None:
    """将 DB pref:routing_overrides 里精确指向 old_model 的条目更新为 new_model。

    DB routing_overrides 优先级高于 lingzhou.json；若不同步，切换主模型后
    重启仍会从 DB 恢复到旧模型，导致 dev model 看似不生效。
    只替换精确等于 old_model 的条目，保留用户有意设置的差异（如 reader: bailian）。
    """
    import sqlite3 as _sqlite3

    if not old_model or not new_model or old_model == new_model:
        return
    try:
        from core.config import Config as _Config

        cfg = _Config.load(cfg_path)
        db_path = cfg.db_path
        if not db_path.exists():
            return
        conn = _sqlite3.connect(str(db_path), timeout=30)
        row = conn.execute(
            "SELECT value FROM facts WHERE key='pref:routing_overrides'"
        ).fetchone()
        if not row:
            conn.close()
            return
        overrides = _json.loads(row[0])
        changed = [tier for tier, model in overrides.items() if model == old_model]
        for tier in changed:
            overrides[tier] = new_model
        if changed:
            conn.execute(
                "UPDATE facts SET value=? WHERE key='pref:routing_overrides'",
                (_json.dumps(overrides, ensure_ascii=False),),
            )
            conn.commit()
            console.print(
                f"[green]✓ DB routing_overrides 已同步:[/green]"
                f" {', '.join(changed)} → [bold cyan]{new_model}[/bold cyan]"
            )
        conn.close()
    except Exception as exc:
        console.print(f"[yellow]⚠ DB routing_overrides 同步失败（非致命）: {exc}[/yellow]")


def _preferred_model_index(catalog_models: list[dict], current_model_id: str = "") -> int:
    """优先当前模型；否则优先 reasoning/thinking 模型；都没有再退回列表首项。"""
    if not catalog_models:
        return -1
    if current_model_id:
        for idx, model in enumerate(catalog_models):
            if str(model.get("id") or "") == current_model_id:
                return idx
    for idx, model in enumerate(catalog_models):
        if model.get("reasoning") or model.get("thinking"):
            return idx
    return 0


def _model_supports_vision(model: dict[str, Any]) -> bool:
    inputs = model.get("input")
    caps = model.get("capabilities")
    return (
        isinstance(inputs, list)
        and "image" in inputs
    ) or (
        isinstance(caps, list)
        and "vision" in caps
    )
