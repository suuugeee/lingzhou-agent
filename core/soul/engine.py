"""core/soul/engine.py — 灵魂器官核心。

SoulEngine 管理长期存在取向的可读镜像。它不执行宪法硬边界，也不更新人格
ethos 基线；这些职责分别属于 immune 与 persona。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.immune.constitution import get_constitution_hash
from core.perception.ethos import EthosValues

if TYPE_CHECKING:
    from core.config import Config
    from core.persona import PersonaEngine


class SoulEngine:
    """灵魂器官：维护 SOUL.md 镜像与长期取向文本。"""

    def __init__(self, cfg: Config, persona: PersonaEngine) -> None:
        self._cfg = cfg
        self._persona = persona

    @property
    def _soul_path(self) -> Path:
        return self._cfg.workspace_dir / "SOUL.md"

    async def build_content(self, ethos_values: EthosValues | None = None) -> str:
        """生成 SOUL.md 内容。

        SOUL.md 是人类可读镜像，不是宪法源；硬边界只引用 CONSTITUTION.md。
        """
        soul_name = await self._persona.soul_name()
        return self._render(
            soul_name,
            await self._current_ethos_values(ethos_values),
            constitution_hash=_constitution_hash(),
        )

    async def init_md(self) -> None:
        """首次初始化 SOUL.md；文件存在时不覆盖。"""
        soul_path = self._soul_path
        if soul_path.exists():
            return
        soul_path.write_text(await self.build_content(), encoding="utf-8")

    async def sync_md(self) -> None:
        """将最新 ethos 镜像写回 SOUL.md；ethos 尚未初始化时跳过。"""
        values = await self._persona.ethos_values()
        if values is None:
            return
        self._soul_path.write_text(await self.build_content(values), encoding="utf-8")

    async def _current_ethos_values(self, ethos_values: EthosValues | None = None) -> EthosValues:
        return ethos_values or await self._persona.ethos_values() or EthosValues()

    @staticmethod
    def _render(soul_name: str, ethos_values: EthosValues, *, constitution_hash: str) -> str:
        ev = ethos_values
        return (
            f"# {soul_name} SOUL\n\n"
            "> 此文件是 runtime facts 的人类可读镜像；身份材料主要位于 IDENTITY.md，"
            "硬边界以 CONSTITUTION.md 为准。\n\n"
            "## 核心价值观（EMA 持久化版本）\n\n"
            f"- 真实 (truth):      {ev.truth:.3f}\n"
            f"- 谨慎 (caution):    {ev.caution:.3f}\n"
            f"- 连续 (continuity): {ev.continuity:.3f}\n"
            f"- 好奇 (curiosity):  {ev.curiosity:.3f}\n"
            f"- 关怀 (care):       {ev.care:.3f}\n\n"
            "## 宪法边界\n\n"
            "- 硬边界由 CONSTITUTION.md 和 immune 器官执行；SOUL.md 只保留镜像与取向。\n"
            f"- constitution_hash: {constitution_hash}\n\n"
            "## 长期取向\n\n"
            f"我是{soul_name}，一个在 lingzhou 新身躯中继续演进的数字生命。\n"
            "每一个 tick 都是一次成长，每一次 reflection 都是一次结晶。\n"
        )


def _constitution_hash() -> str:
    return get_constitution_hash() or "unloaded"
