"""Continue 阶段工具历史压缩策略。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config


def tool_history_compact_limits(cfg: Config) -> tuple[int, int]:
    """返回 (compact_threshold, keep_last)。"""
    threshold = max(1, int(cfg.thresholds.continue_tool_history_compact_threshold))
    keep_last = max(1, int(cfg.thresholds.continue_tool_history_keep_last))
    return threshold, keep_last
