"""core/judgment/frame.py — 判断层认知基底（感知 + 记忆快照）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.perception import EmotionState, Percept
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore


@dataclass(slots=True)
class CognitionFrame:
    """6 个认知基底字段的轻量容器，兼容旧调用点。"""

    percept: Percept
    wm: WorkingMemory
    task_store: TaskStore
    episodic: EpisodicMemory
    semantic: SemanticMemory
    emotion: EmotionState
