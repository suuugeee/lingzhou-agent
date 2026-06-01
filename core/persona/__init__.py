"""core.persona — 人格层（Soul 工作区文件、SelfModel、PersonaEngine）。"""
from __future__ import annotations

from .engine import PersonaEngine
from .self_model import SelfModel, fmt_self_model
from .soul import SoulManager

__all__ = ["PersonaEngine", "SoulManager", "SelfModel", "fmt_self_model"]
