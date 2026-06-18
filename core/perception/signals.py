"""core/perception/signals.py — 判断信号 + 认知信号。

JudgmentSignals：LLM 调用前的确定性预判姿态。
CognitiveSignals：完整认知状态报告，注入 LLM 判断上下文。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.perception.emotion import EmotionState


def _clean_signal_text(text: str) -> str:
    return " ".join((text or "").split())


# ── 判断信号 ──────────────────────────────────────────────────────────────────

@dataclass
class JudgmentSignals:
    """在 LLM 调用前确定性推导的判断信号（姿态）。"""
    require_more_evidence: bool = False
    prefer_narrow_scope: bool = False
    posture: str = "act"    # act | pause | narrow


def compute_judgment_signals(
    failure_count: int,
    high_error_streak: int,
    perception_trend: str,
    emotion_state: EmotionState,
    thresholds: Any,
) -> JudgmentSignals:
    """LLM 前的确定性预判：减少"冷启动"时 LLM 从零估算的不确定性。"""
    error_streak_guard = max(1, int(thresholds.judgment_error_streak_guard))
    worsening_evidence_failure_count = max(
        1,
        int(thresholds.judgment_require_more_evidence_worsening_failure_count),
    )
    prefer_narrow_failure_count = max(
        1,
        int(thresholds.judgment_prefer_narrow_failure_count),
    )
    posture_narrow_failure_count = max(
        1,
        int(thresholds.judgment_posture_narrow_failure_count),
    )
    posture_narrow_down_regulate_failure_count = max(
        1,
        int(thresholds.judgment_posture_narrow_down_regulate_failure_count),
    )
    posture_pause_worsening_failure_count = max(
        1,
        int(thresholds.judgment_posture_pause_worsening_failure_count),
    )
    sig = JudgmentSignals()
    if high_error_streak >= error_streak_guard or (
        perception_trend == "worsening" and failure_count >= worsening_evidence_failure_count
    ):
        sig.require_more_evidence = True
    if failure_count >= prefer_narrow_failure_count or high_error_streak >= error_streak_guard:
        sig.prefer_narrow_scope = True
    if failure_count >= posture_narrow_failure_count or (
        failure_count >= posture_narrow_down_regulate_failure_count
        and emotion_state.regulation.strategy == "down-regulate"
    ):
        sig.posture = "narrow"
    elif high_error_streak >= error_streak_guard or (
        perception_trend == "worsening" and failure_count >= posture_pause_worsening_failure_count
    ):
        sig.posture = "pause"
    return sig


# ── 认知信号 ──────────────────────────────────────────────────────────────────
# 设计原则：此类只报告信号强度，不产生任何决策或任务文字。
# 是否 task.add、如何命名任务、如何响应异常，全部由 LLM 在 judgment 层决定。

@dataclass
class CognitiveSignals:
    """感知层推导的认知状态信号，注入 LLM 判断上下文。"""
    emotion_activation: float = 0.0
    emotion_alert: bool = False             # 激活超阈值
    wm_pressure: float = 0.0
    wm_pressure_alert: bool = False         # WM 压力超阈值
    prediction_error: float = 0.0
    prediction_error_alert: bool = False    # 预测误差超阈值
    has_active_task: bool = False
    idle_cycles: int = 0                    # 无活跃任务持续轮次
    next_step_fulfilled: bool | None = None  # 上轮 next_step 是否被执行（None=首轮）
    # 循环探针（由 loop 注入）：给 LLM 的结构化反循环信号
    repeat_action_count: int = 0
    repeat_action_tool: str = ""
    repeat_action_key: str = ""
    repeat_read_count: int = 0
    repeat_read_path: str = ""
    repeat_list_count: int = 0
    repeat_list_path: str = ""
    # 探索预算感知
    explore_count: int = 0
    explore_threshold: int = 0
    loop_probe_version: int = 0
    # 上一动作结果
    last_action_tool: str = ""
    last_action_key: str = ""
    last_action_status: str = ""
    last_action_summary: str = ""
    last_action_error: str = ""
    last_action_state_delta: str = ""
    last_action_progressful: bool | None = None
    last_action_progress_reason: str = ""
    recent_action_history: list[str] = field(default_factory=list[str])

    def to_text(self) -> str:
        """格式化为 LLM 可读文本，注入 judgment bundle。"""
        lines: list[str] = []
        lines.append(
            "loop_probe="
            f"{{version={self.loop_probe_version}, "
            f"repeat_action_count={self.repeat_action_count}, "
            f"repeat_action_tool='{self.repeat_action_tool}', "
            f"repeat_action_key='{self.repeat_action_key}', "
            f"repeat_read_count={self.repeat_read_count}, "
            f"repeat_read_path='{self.repeat_read_path}', "
            f"repeat_list_count={self.repeat_list_count}, "
            f"repeat_list_path='{self.repeat_list_path}'}}"
        )
        if self.last_action_tool or self.last_action_status:
            lines.append(
                "last_action="
                f"{{tool='{self.last_action_tool}', "
                f"key='{self.last_action_key}', "
                f"status='{self.last_action_status}', "
                f"progressful={self.last_action_progressful}, "
                f"error='{_clean_signal_text(self.last_action_error)}', "
                f"state_delta='{_clean_signal_text(self.last_action_state_delta)}', "
                f"summary='{_clean_signal_text(self.last_action_summary)}'}}"
            )
        if self.recent_action_history:
            lines.append("recent_actions:")
            lines.extend(f"- {_clean_signal_text(item)}" for item in self.recent_action_history)
        if self.repeat_action_count >= 3 and self.repeat_action_tool:
            lines.append(
                f"⚠️ 最近动作已连续重复 {self.repeat_action_count} 次："
                f"tool={self.repeat_action_tool} key={self.repeat_action_key or '（空）'}。"
                "请结合最近结果、错误与参数判断是否还值得继续。"
            )
        if self.repeat_read_count >= 3 and self.repeat_read_path:
            lines.append(
                f"⚠️ 最近读取已连续命中相同内容 {self.repeat_read_count} 次：{self.repeat_read_path}。"
                "若没有新的外部变化，继续读取大概率只会重复。"
            )
        if self.repeat_list_count >= 3 and self.repeat_list_path:
            lines.append(
                f"⚠️ 最近目录枚举已连续命中相同结果 {self.repeat_list_count} 次：{self.repeat_list_path}。"
                "请先判断是否需要切换到读取、写入或总结。"
            )
        if self.explore_count >= self.explore_threshold and self.explore_threshold > 0:
            lines.append(
                f"⚠️ 当前任务已探索 {self.explore_count} 次（阈值 {self.explore_threshold}），请评估是否已有足够信息推进"
            )
        if self.last_action_tool and self.last_action_progressful is False and self.last_action_status in {"ok", "skipped"}:
            reason = f"（原因: {self.last_action_progress_reason}）" if self.last_action_progress_reason else ""
            lines.append(
                f"⚠️ 上一轮动作 {self.last_action_tool} 被系统判定为未推进。{reason}\n"
                "这不一定意味着你做错了——系统判断基于工具类型和结果指纹。"
                "如果你认为已有足够信息推进任务，可以在 rationale 中说明并继续。"
            )
        if self.emotion_alert:
            lines.append(
                f"⚠️ 情绪激活偏高（{self.emotion_activation:.2f}）："
                "可能处于压力或亢奋状态，建议自检或放缓节奏"
            )
        if self.wm_pressure_alert:
            lines.append(
                f"⚠️ 工作记忆压力偏高（{self.wm_pressure:.0%}）"
                "  请先调用 memory.snapshot（快照并清空 WM），"
                "再视情况用 reflect.structural 提炼洞察"
            )
        if self.prediction_error_alert:
            lines.append(
                f"⚠️ 预测误差偏高（{self.prediction_error:.2f}）："
                "环境或任务状态超出预期，建议重新评估"
            )
        if not self.has_active_task:
            lines.append(f"ℹ️ 当前无活跃任务（已空转 {self.idle_cycles} 轮）")
            if self.idle_cycles >= 2:
                lines.append(
                    '→ 建议：使用 task.add 创建一个自驱任务，'
                    '例如"建立环境认知地图"或"初始化自我状态检查"'
                )
        if self.next_step_fulfilled is False:
            lines.append(
                "⚠️ 上一轮计划的 next_step 未真正推进（可能是 wait/pause，或 act 但没有产生新结果），"
                "注意避免计划漂移"
            )
        if not lines:
            lines.append("✓ 认知状态正常，无异常信号")
        return "\n".join(lines)
