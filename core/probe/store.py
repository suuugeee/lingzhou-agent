"""core/probe/store.py — 探针配置持久化（JSON 文件）。

完全解耦于 lingzhou 主数据库：探针配置保存在工作区 probes.json 文件中。
方法均为 async 以保持调用方签名不变，实际为同步内存操作 + 文件写入。
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from core.contracts.probe import (
    ProbeConfig,
    ProbeDataBack,
    ProbeKind,
    normalize_probe_coverage_tags,
)

if TYPE_CHECKING:
    from pathlib import Path

_log = logging.getLogger("lingzhou.probe")


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_dict(cfg: ProbeConfig) -> dict[str, Any]:
    return {
        "id": cfg.id,
        "name": cfg.name,
        "kind": cfg.kind,
        "spec": cfg.spec,
        "trigger": cfg.trigger,
        "purpose": cfg.purpose,
        "data_back": cfg.data_back,
        "coverage_tags": list(cfg.coverage_tags),
        "alert_expr": cfg.alert_expr,
        "alert_message": cfg.alert_message,
        "enabled": cfg.enabled,
        "created_at": cfg.created_at,
        "last_run_at": cfg.last_run_at,
        "last_result": cfg.last_result,
        "last_error": cfg.last_error,
        "last_confidence": cfg.last_confidence,
        "last_confidence_reason": cfg.last_confidence_reason,
        "last_suspect": cfg.last_suspect,
        "last_alerted": cfg.last_alerted,
        "last_alert_detail": cfg.last_alert_detail,
    }


def _from_dict(d: dict[str, Any]) -> ProbeConfig:
    data_back_raw = str(d.get("data_back") or "wm")
    # 兼容旧数据：chat 模式已废弃，降级为 wm
    if data_back_raw not in ("none", "wm"):
        data_back_raw = "wm"
    kind_raw = str(d.get("kind") or "shell")
    if kind_raw not in ("shell", "http", "python", "builtin"):
        kind_raw = "shell"
    return ProbeConfig(
        id=int(d.get("id") or 0),
        name=str(d["name"]),
        kind=cast("ProbeKind", kind_raw),
        spec=str(d.get("spec", "")),
        trigger=str(d.get("trigger", "manual")),
        purpose=str(d.get("purpose") or ""),
        data_back=cast("ProbeDataBack", data_back_raw),
        coverage_tags=normalize_probe_coverage_tags(d.get("coverage_tags") or []),
        alert_expr=d.get("alert_expr") or None,
        alert_message=d.get("alert_message") or None,
        enabled=bool(d.get("enabled", True)),
        created_at=str(d.get("created_at", "")),
        last_run_at=d.get("last_run_at") or None,
        last_result=d.get("last_result") or None,
        last_error=d.get("last_error") or None,
        last_confidence=_as_optional_float(d.get("last_confidence")),
        last_confidence_reason=d.get("last_confidence_reason") or None,
        last_suspect=bool(d.get("last_suspect", False)),
        last_alerted=bool(d.get("last_alerted", False)),
        last_alert_detail=d.get("last_alert_detail") or None,
    )


class ProbeStore:
    """探针配置 CRUD。使用 JSON 文件持久化，与 lingzhou 主 DB 完全解耦。"""

    def __init__(self, probe_file: Path) -> None:
        self._file = probe_file
        self._probes: dict[str, ProbeConfig] = {}
        self._next_id: int = 1

    def load(self) -> None:
        """从 JSON 文件加载探针配置（同步，在事件循环启动前调用）。"""
        if not self._file.exists():
            return
        try:
            entries = json.loads(self._file.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                entries = []
            for d in entries:
                try:
                    cfg = _from_dict(d)
                    self._probes[cfg.name] = cfg
                    if cfg.id >= self._next_id:
                        self._next_id = cfg.id + 1
                except Exception as exc:
                    _log.warning("[probe] 跳过无效探针配置项: %s", exc)
            _log.info("[probe] 已加载 %d 个探针 (%s)", len(self._probes), self._file.name)
        except Exception as exc:
            _log.warning("[probe] 读取 %s 失败: %s", self._file, exc)

    def _save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            data = [_to_dict(p) for p in self._probes.values()]
            self._file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            _log.warning("[probe] 写入 %s 失败: %s", self._file, exc)

    async def upsert(self, cfg: ProbeConfig) -> int:
        existing = self._probes.get(cfg.name)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        saved = ProbeConfig(
            id=existing.id if existing else self._next_id,
            name=cfg.name,
            kind=cfg.kind,
            spec=cfg.spec,
            trigger=cfg.trigger,
            purpose=cfg.purpose,
            data_back=cfg.data_back,
            coverage_tags=normalize_probe_coverage_tags(cfg.coverage_tags),
            alert_expr=cfg.alert_expr,
            alert_message=cfg.alert_message,
            enabled=cfg.enabled,
            created_at=existing.created_at if existing else now,
            last_run_at=existing.last_run_at if existing else None,
            last_result=existing.last_result if existing else None,
            last_error=existing.last_error if existing else None,
            last_confidence=existing.last_confidence if existing else None,
            last_confidence_reason=existing.last_confidence_reason if existing else None,
            last_suspect=existing.last_suspect if existing else False,
            last_alerted=existing.last_alerted if existing else False,
            last_alert_detail=existing.last_alert_detail if existing else None,
        )
        if not existing:
            self._next_id += 1
        self._probes[cfg.name] = saved
        self._save()
        return saved.id

    async def delete(self, name: str) -> bool:
        if name not in self._probes:
            return False
        del self._probes[name]
        self._save()
        return True

    async def get(self, name: str) -> ProbeConfig | None:
        return self._probes.get(name)

    async def list_all(self, enabled_only: bool = False) -> list[ProbeConfig]:
        probes = list(self._probes.values())
        if enabled_only:
            probes = [p for p in probes if p.enabled]
        return probes

    async def update_run_result(
        self,
        name: str,
        last_run_at: str,
        last_result: str | None,
        last_error: str | None,
        last_confidence: float | None = None,
        last_confidence_reason: str | None = None,
        last_suspect: bool = False,
        last_alerted: bool = False,
        last_alert_detail: str | None = None,
    ) -> None:
        cfg = self._probes.get(name)
        if cfg is None:
            return
        self._probes[name] = ProbeConfig(
            id=cfg.id,
            name=cfg.name,
            kind=cfg.kind,
            spec=cfg.spec,
            trigger=cfg.trigger,
            purpose=cfg.purpose,
            data_back=cfg.data_back,
            coverage_tags=cfg.coverage_tags,
            alert_expr=cfg.alert_expr,
            alert_message=cfg.alert_message,
            enabled=cfg.enabled,
            created_at=cfg.created_at,
            last_run_at=last_run_at,
            last_result=last_result,
            last_error=last_error,
            last_confidence=last_confidence,
            last_confidence_reason=last_confidence_reason,
            last_suspect=last_suspect,
            last_alerted=last_alerted,
            last_alert_detail=last_alert_detail,
        )
        self._save()

    async def set_enabled(self, name: str, enabled: bool) -> bool:
        cfg = self._probes.get(name)
        if cfg is None:
            return False
        self._probes[name] = ProbeConfig(
            id=cfg.id,
            name=cfg.name,
            kind=cfg.kind,
            spec=cfg.spec,
            trigger=cfg.trigger,
            purpose=cfg.purpose,
            data_back=cfg.data_back,
            coverage_tags=cfg.coverage_tags,
            alert_expr=cfg.alert_expr,
            alert_message=cfg.alert_message,
            enabled=enabled,
            created_at=cfg.created_at,
            last_run_at=cfg.last_run_at,
            last_result=cfg.last_result,
            last_error=cfg.last_error,
            last_confidence=cfg.last_confidence,
            last_confidence_reason=cfg.last_confidence_reason,
            last_suspect=cfg.last_suspect,
            last_alerted=cfg.last_alerted,
            last_alert_detail=cfg.last_alert_detail,
        )
        self._save()
        return True
