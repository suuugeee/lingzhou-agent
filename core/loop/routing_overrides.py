"""Helpers for runtime routing override persistence and validation."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from core.judgment.tiers import JUDGMENT_TIER_SET
from core.metabolic.lifecycle_utils import _decision_basis_from_parts as _decision_basis


def _valid_model_ref(model_ref: str, *, catalog_path: Path | None = None) -> bool:
    provider, sep, model_id = str(model_ref or "").partition("/")
    if not provider or not sep or not model_id:
        return False
    if model_id.strip().isdigit():
        return False
    try:
        from provider.catalog import lookup_model_ref

        return lookup_model_ref(model_ref, catalog_path=catalog_path) is not None
    except Exception:
        return True


def normalize_routing_overrides(payload: Any, *, catalog_path: Path | None = None) -> dict[str, str] | None:
    """Return validated tier -> model overrides, accepting the legacy flat JSON shape."""
    if not isinstance(payload, dict):
        return None
    raw_overrides = payload.get("overrides") if isinstance(payload.get("overrides"), dict) else payload
    overrides = {
        str(tier): str(model_ref).strip()
        for tier, model_ref in raw_overrides.items()
        if tier in JUDGMENT_TIER_SET
        and isinstance(model_ref, str)
        and _valid_model_ref(model_ref, catalog_path=catalog_path)
    }
    return overrides or None


def normalize_routing_overrides_payload(
    payload: Any,
    *,
    catalog_path: Path | None = None,
) -> dict[str, str] | None:
    """兼容字符串/对象形式的路由覆盖持久化内容。"""
    if payload is None:
        return None
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    return normalize_routing_overrides(parsed, catalog_path=catalog_path)


def routing_overrides_meta(*, source: str, decision_basis: str = "") -> dict[str, str]:
    meta = {"source": str(source or "unknown")}
    basis = _decision_basis(decision_basis)
    if basis:
        meta["decision_basis"] = basis
    return meta
