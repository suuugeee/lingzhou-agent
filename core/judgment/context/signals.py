"""core/judgment/context/signals.py — 判断信号、边界与感知重放格式化。"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.perception import CognitiveSignals, JudgmentSignals, PerceptionReplaySummary


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


def _fmt_risk_sections(
    judgment_signals: JudgmentSignals | None,
    failures: list[object],
    durable_failure_snapshot: dict[str, object] | None,
    perception_replay: PerceptionReplaySummary | None,
    cognitive_signals: CognitiveSignals | None,
) -> str:
    lines: list[str] = []
    if judgment_signals:
        if judgment_signals.posture == "pause":
            lines.append("⚠️ 姿态层建议暂停，建议先验证是否有新增可置信证据。")
        elif judgment_signals.posture == "narrow":
            lines.append("⚠️ 姿态层建议收窄决策范围，优先选择低步幅动作。")
        if judgment_signals.require_more_evidence:
            lines.append("⚠️ 当前信号已触发 require_more_evidence：存在证据不足。")
        if judgment_signals.prefer_narrow_scope:
            lines.append("⚠️ 当前建议倾向 narrow scope，非关键分支先降优先级。")

    if failures:
        lines.append(f"- 最近失败可见风险: {len(failures)} 条，优先处理最近高频失效路径。")

    if durable_failure_snapshot:
        muted = durable_failure_snapshot.get("muted_actions") or []
        if muted:
            for item in muted[:5]:
                if not isinstance(item, dict):
                    continue
                tool = str(item.get("tool") or "-")
                key = str(item.get("key") or "")
                reason = str(item.get("reason") or "stable_failure")
                count = int(item.get("count") or 0)
                remaining = int(item.get("remaining_sec") or 0)
                marker = f"tool={tool}"
                if key:
                    marker += f" key={key}"
                lines.append(f"- 稳定失败约束: {marker}（{reason}, count={count}, 剩余 {remaining}s）")

    if perception_replay and perception_replay.trend == "worsening":
        lines.append(
            f"⚠️ 感知趋势恶化：样本 {perception_replay.samples}，"
            f"连续高误差 {perception_replay.high_error_streak}，偏离路径可能增大。"
        )

    if cognitive_signals:
        if cognitive_signals.wm_pressure_alert:
            lines.append(f"⚠️ WM 压力偏高（{cognitive_signals.wm_pressure:.0%}），高频动作可能放大失真。")
        if cognitive_signals.repeat_action_count >= 2:
            lines.append(
                f"⚠️ 行为重复风险：tool={cognitive_signals.repeat_action_tool or 'unknown'} "
                f"重复 {cognitive_signals.repeat_action_count} 次，若无新证据应切动作或证据源。"
            )
        if cognitive_signals.repeat_read_count >= 3 and cognitive_signals.repeat_read_path:
            lines.append(f"⚠️ 重复读取 {cognitive_signals.repeat_read_path}（{cognitive_signals.repeat_read_count} 次）可能进入低增量循环。")
        if cognitive_signals.repeat_list_count >= 3 and cognitive_signals.repeat_list_path:
            lines.append(f"⚠️ 重复列表枚举 {cognitive_signals.repeat_list_path}（{cognitive_signals.repeat_list_count} 次）可能收益递减。")

    return "\n".join(lines) if lines else "（当前未识别到高风险项）"


def _fmt_uncertainty_sections(
    judgment_signals: JudgmentSignals | None,
    perception_replay: PerceptionReplaySummary | None,
    cognitive_signals: CognitiveSignals | None,
) -> str:
    lines: list[str] = []
    if judgment_signals and judgment_signals.require_more_evidence:
        lines.append("证据缺口：require_more_evidence=True，仍缺少可直接驱动下一步的高可信证据。")
    if perception_replay:
        if perception_replay.samples < 4:
            lines.append(f"观测样本偏少（samples={perception_replay.samples}），趋势仅是近似判断。")
        elif perception_replay.trend == "worsening":
            lines.append("观测不确定性：趋势恶化，需外部输入/新的工具路径确认方向。")
        elif perception_replay.trend == "insufficient_data":
            lines.append("观测不确定性：重放样本不足，趋势标签不足以作为决策主证据。")
        if perception_replay.high_error_streak >= 3:
            lines.append("观测不确定性：高误差连续段偏长，建议先确认环境状态再推进。")
    if cognitive_signals:
        if cognitive_signals.next_step_fulfilled is False:
            lines.append("路径不确定：上轮 next_step 未真实推进，当前 step 是否仍对应目标存在漂移。")
        if not cognitive_signals.has_active_task:
            lines.append("当前无活跃任务：是继续探索、暂停确认，还是等待外部输入，需要先明确触发条件。")
        if cognitive_signals.last_action_progressful is False and cognitive_signals.last_action_status in {"ok", "skipped"}:
            lines.append(f"最新动作 {cognitive_signals.last_action_tool or 'unknown'} 未推进，是否仅是确认性动作需再判定。")
    return "\n".join(lines) if lines else "（当前不确定性可接受）"


_WM_SECTION_PAT = re.compile(r"^(?P<name>[A-Za-z0-9_\u4e00-\u9fff]+):\s*(?P<tail>.*)$")
_NO_WM_PROPOSAL_SECTIONS = "（WM 中未发现可执行提案区块）"


def _extract_wm_block(content: str, marker: str) -> list[str]:
    """提取 WM 文本中的 `marker:` 块内容。

    块定义为以 marker 为前缀的行开始，到下一个 section-header 行结束。
    """
    if not content:
        return []
    lines = content.splitlines()
    collecting = False
    raw: list[str] = []
    marker_prefix = marker + ":"

    for line in lines:
        stripped = line.strip()
        if not collecting:
            if stripped.startswith(marker_prefix):
                collecting = True
                suffix = stripped[len(marker_prefix) :].strip()
                if suffix:
                    raw.append(suffix)
            continue

        if _WM_SECTION_PAT.match(stripped):
            break
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("• "):
            raw.append(stripped[2:].strip())
        else:
            raw.append(stripped)

    return [item for item in raw if item]


def _fmt_wm_proposal_sections(wm_items: list[dict[str, object]]) -> str:
    """从 WM 的 observation 信号里抽取可执行建议与方向。

    目的：让 judgment 上下文不仅看到内容，还能在“可落地动作”层面给出可决策候选。
    """
    if not wm_items:
        return _NO_WM_PROPOSAL_SECTIONS

    lines: list[str] = []
    max_items = 3
    max_proposals = 4
    max_questions = 3
    max_directions = 4
    count = 0

    for item in wm_items:
        if count >= max_items:
            break
        content = str(item.get("content") or "")
        kind = str(item.get("kind") or "unknown")
        proposals = _extract_wm_block(content, "proposal")
        open_questions = _extract_wm_block(content, "open_questions")
        available_directions = _extract_wm_block(content, "available_directions")
        if not (proposals or open_questions or available_directions):
            continue

        scope_match = re.search(r"^scope:\s*(.*)$", content, flags=re.MULTILINE)
        scope = (scope_match.group(1).strip() if scope_match else "unknown")
        header = f"- [{kind}] scope={scope}"

        if proposals:
            header += " | proposal:"
            lines.append(header)
            lines.extend([f"  - {line}" for line in proposals[:max_proposals]])
        else:
            lines.append(header)

        if open_questions:
            lines.append("  open_questions:")
            lines.extend([f"    - {line}" for line in open_questions[:max_questions]])
        if available_directions:
            normalized_dir = []
            for raw_dir in available_directions:
                for token in re.split(r"[|,；;]", raw_dir):
                    text = token.strip()
                    if text:
                        normalized_dir.append(text)
            if normalized_dir:
                lines.append("  available_directions:")
                lines.extend([f"    - {text}" for text in normalized_dir[:max_directions]])
        count += 1

    if not lines:
        return _NO_WM_PROPOSAL_SECTIONS
    return "\n".join(lines)
