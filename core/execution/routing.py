"""core.execution.routing — execution 层路由装配辅助。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.execution.run_profile import RUN_TYPE_DEFAULT_TIER


def _coerce_run_type_override(candidate: Any) -> dict[str, str] | None:
    """标准化 run_type_routing 覆盖值。

    返回 None 表示无有效覆盖（包括非 Mapping 的输入），否则返回仅保留字符串键值的纯映射。
    """
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        return None
    sanitized: dict[str, str] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip().lower()
        if normalized_key and normalized_value:
            sanitized[normalized_key] = normalized_value
    if not sanitized:
        return None
    return sanitized


def _assemble_run_type_routing(cfg: dict[str, str] | None) -> dict[str, str]:
    """聚合内置默认、catalog 与配置覆盖，返回最终有效 mapping。"""
    catalog = _coerce_run_type_override(_load_catalog_run_type_routing()) or {}
    base: dict[str, str] = dict(RUN_TYPE_DEFAULT_TIER)
    base.update(catalog)
    if cfg:
        base.update(cfg)
    return base


def _load_catalog_run_type_routing() -> dict[str, str] | None:
    from provider.catalog import get_run_type_routing

    try:
        return get_run_type_routing()
    except Exception:
        return None


def resolve_run_type_routing(cfg: Any | None = None) -> dict[str, str]:
    """从配置加载并返回 run_type→tier 的完整映射（含内置兜底和 catalog 覆盖）。"""
    raw_cfg = _coerce_run_type_override(
        cfg.get("run_type_routing") if isinstance(cfg, dict) else getattr(cfg, "run_type_routing", None)
    )
    return _assemble_run_type_routing(raw_cfg)
