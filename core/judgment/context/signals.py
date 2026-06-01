"""core/judgment/context/signals.py — 判断信号、边界与感知重放格式化。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.perception import JudgmentSignals, PerceptionReplaySummary


def _fmt_judgment_signals(signals: JudgmentSignals | None) -> str:
    if not signals:
        return "（JudgmentSignals 未计算）"
    return (
        f"posture={signals.posture}  "
        f"require_more_evidence={signals.require_more_evidence}  "
        f"prefer_narrow_scope={signals.prefer_narrow_scope}"
    )


def _fmt_hard_boundaries(hard_boundaries: list[str] | None) -> str:
    if not hard_boundaries:
        return "（无 hard_boundary 限制）"
    return "\n".join(f"- {boundary}" for boundary in hard_boundaries)


def _fmt_perception_replay(replay: PerceptionReplaySummary | None) -> str:
    if not replay:
        return "（感知重放不可用）"
    lines = [
        f"样本数={replay.samples}  平均预测误差={replay.avg_prediction_error:.2f}  连续高误差={replay.high_error_streak}  趋势={replay.trend}",
    ]
    if replay.hints:
        lines.extend(f"提示: {hint}" for hint in replay.hints)
    return "\n".join(lines)
