"""core/persona/engine.py — PersonaEngine：人格器官核心。

管理 ethos 基线读取与强类型转换。SOUL.md 镜像由 core.soul.SoulEngine 负责；
宪法硬边界由 core.immune 负责。
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import Config
    from core.perception.ethos import EthosValues
    from store.task import TaskStore

_log = logging.getLogger("lingzhou.persona")


def _loads_json_dict(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


class PersonaEngine:
    """人格器官：管理长期气质与行为倾向的 ethos 基线。"""

    def __init__(self, cfg: Config, task_store: TaskStore) -> None:
        self._cfg = cfg
        self._task_store = task_store

    async def soul_name(self) -> str:
        """从 facts DB 读取 soul:name，回退到 cfg.soul.name。"""
        name_val, name_found = await self._task_store.get_fact("soul:name")
        return name_val if name_found and name_val else self._cfg.soul.name

    async def ethos_from_db(self) -> dict[str, Any]:
        """从 facts DB 读取 soul:ethos_baseline，解析失败返回空 dict。"""
        ethos_json, found = await self._task_store.get_fact("soul:ethos_baseline")
        if not found or not ethos_json:
            return {}
        return _loads_json_dict(ethos_json)

    async def ethos_values(self) -> EthosValues | None:
        """读取并解析 ethos_baseline；尚未初始化时返回 None。"""
        from core.perception.ethos import EthosValues

        ethos_raw = await self.ethos_from_db()
        if not ethos_raw:
            return None
        try:
            return EthosValues.from_dict(ethos_raw)
        except ValueError as exc:
            _log.warning("[persona] ethos_baseline 解析失败: %s", exc)
            return None
