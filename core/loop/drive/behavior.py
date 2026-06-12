"""core/loop/drive/behavior.py — 行为模式追踪与认知信号生成。

检测 agent 是否陷入重复行为，并生成 WM 自我感知条目供 LLM 自主判断：
  1. action streak：连续 3 次完全相同的 (tool, key_param) 对
  2. read streak：连续 3 次 file.read 读取相同文件的相同内容（按内容 MD5 去重）
  3. rationale 指纹：连续相同推理结论超阈值时触发信念固化警告

检测结果优先以 WMItem 形式注入工作记忆；当同一低增量动作已明确形成
连续循环时，执行门控会把动作改道到总结/换策略，避免继续消耗工具轮次。
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from core.contracts.execution import action_key_param
from core.judgment.output import JudgmentOutput
from tools.registry import tool_has_capability

if TYPE_CHECKING:
    from memory.working import WMItem

_log = logging.getLogger("lingzhou.behavior_tracker")

# 同文件连续窗口探测阈值：同一文件被分窗口读取超过此次数时触发特定警告
_SEQ_WINDOW_WARN_AT = 3
# 窗口连续性判断：相邻读的 start 与上一次 end 的差距在此比例内视为连续
_SEQ_WINDOW_GAP_RATIO = 0.25  # 25% 窗口大小内视作连续

# rationale 指纹：连续相同结论超过此阈值时触发"信念固化"警告
_BELIEF_STALE_THRESHOLD = 4
# rationale 指纹窗口大小（deque maxlen)
_BELIEF_WINDOW = 8
_LOW_INCREMENT_EXPLORATION_TOOLS = frozenset({
    "file.list",
    "memory.search",
    "task.list",
    "probe.list",
    "config.list_keys",
})
_LOW_INCREMENT_TOOL_GROUPS = {
    "file.list": "inventory",
    "task.list": "inventory",
    "probe.list": "inventory",
    "config.list_keys": "inventory",
    "memory.search": "memory_search",
}
_WAIT_MARKERS = (
    "等待",
    "外部输入",
    "外部信号",
    "下一次",
    "到期",
    "用户输入",
    "wait ",
    "external",
    "signal",
    "inbox",
)
_EVIDENCE_MARKERS = (
    "读取",
    "查询",
    "检查",
    "验证",
    "定位",
    "列出",
    "分析",
    "运行",
    "执行",
    "修复",
    "实现",
    "测试",
    "日志",
    "schema",
    "read",
    "query",
    "check",
    "verify",
    "inspect",
    "list",
    "analyze",
    "run",
    "execute",
    "fix",
    "implement",
    "test",
    "log",
)


class BehaviorTracker:
    """行为模式追踪器：检测循环并把信号交给 LLM。"""

    def __init__(
        self,
        wait_streak_notify: list[int] | None = None,
        streak_threshold: int = 3,
        wm_priorities: dict[str, float] | None = None,
        registry: Any = None,
        seq_window_warn_at: int = _SEQ_WINDOW_WARN_AT,
        seq_window_gap_ratio: float = _SEQ_WINDOW_GAP_RATIO,
        belief_stale_threshold: int = _BELIEF_STALE_THRESHOLD,
        belief_window: int = _BELIEF_WINDOW,
    ) -> None:
        # wait-streak 通知阈值（升序，来自配置；None → 使用默认 [3, 6]）
        self._wait_notify_thresholds: list[int] = sorted(wait_streak_notify) if wait_streak_notify else [3, 6]
        self._streak_threshold: int = max(1, streak_threshold)
        self._recent_actions: deque[tuple[str, str]] = deque(maxlen=self._streak_threshold)
        self._recent_read_fps: deque[tuple[str, int, str]] = deque(maxlen=self._streak_threshold)
        self._recent_list_fps: deque[tuple[str, str]] = deque(maxlen=self._streak_threshold)
        self._action_streak_sig: tuple[str, str] | None = None
        self._action_streak_count: int = 0
        self._read_streak_fp: tuple[str, int, str] | None = None
        self._read_streak_count: int = 0
        self._list_streak_fp: tuple[str, str] | None = None
        self._list_streak_count: int = 0
        self._loop_probe_version: int = 0
        # 顺序窗口和信念指纹探测参数（来自 ThresholdsConfig）
        self._seq_window_warn_at: int = max(1, seq_window_warn_at)
        self._seq_window_gap_ratio: float = seq_window_gap_ratio
        self._belief_stale_threshold: int = max(1, belief_stale_threshold)
        # 同文件顺序窗口探测追踪
        self._seq_window_path: str | None = None
        self._seq_window_count: int = 0
        self._seq_window_last_end: int = 0
        self._seq_window_warned: bool = False
        self._wait_streak: int = 0          # 连续 wait/pause 决策次数
        self._wait_streak_warned: set[int] = set()   # 已触发通知的阈值
        # rationale 指纹追踪（信念固化检测）
        self._rationale_hashes: deque[str] = deque(maxlen=max(1, belief_window))
        self._belief_stale_hash: str | None = None
        self._belief_stale_count: int = 0
        self._belief_stale_warned: bool = False
        # 上次 act 结果指纹（用于 on_act_result 折回调用）
        self._last_act_result_fp: str = ""
        self._edit_fail_count: int = 0
        # WM 优先级（可由外部通过 wm_priorities 注入，确保所有值来自 ThresholdsConfig）
        _pri = wm_priorities or {}
        self._pri_behavior_loop: float = float(_pri.get("behavior_loop", 0.95))
        self._pri_edit_caution: float = float(_pri.get("edit_caution", 0.93))
        self._pri_belief_stale: float = float(_pri.get("belief_stale", 0.96))
        self._registry = registry
    @property
    def wait_streak(self) -> int:
        """公开接口：连续 wait/pause 决策次数。"""
        return self._wait_streak

    @property
    def read_streak_count(self) -> int:
        """公开接口：连续读取相同内容的次数。"""
        return self._read_streak_count
    @property
    def list_streak_count(self) -> int:
        """公开接口：连续列出相同目录结果的次数。"""
        return self._list_streak_count

    # ── 状态接口 ──────────────────────────────────────────────────────────────

    def apply_execution_gate(self, action: JudgmentOutput, signals: Any) -> JudgmentOutput:
        """在明确重复同一低增量动作时硬制动，避免继续消耗上下文与工具轮次。"""
        if self._should_gate_evidence_task_wait(action, signals):
            task_id = str(getattr(signals, "active_task_id", "") or "").strip()
            next_step = str(getattr(signals, "active_task_next_step", "") or "").strip()
            return self._build_workbench_gate(
                action=action,
                domain="self-drive-evidence",
                intent="阻止自驱取证任务在证据未产生前进入等待",
                evidence=[
                    f"task#{task_id or '?'} 是自驱任务，next_step 明确要求取证。",
                    f"本轮模型选择了 {action.decision}，但尚未产生 next_step 所需的新证据。",
                ],
                hypothesis="当前卡点不是缺少外部信号，而是自驱任务把可执行的取证步骤误判成了等待条件。",
                next_verification=next_step,
                completion_checks=[
                    "已执行 next_step 指向的低风险取证动作。",
                    "已把取证结果写入 task.workbench/current_step/result summary。",
                    "只有在外部依赖明确时才进入 task.wait。",
                ],
                rationale=(
                    "行为门控改道：自驱任务仍有明确取证 next_step，"
                    f"不能直接 {action.decision}，先写入恢复工作台约束下一轮行动。"
                ),
                next_step=next_step,
                recovery_state="evidence_required_before_wait",
            )

        repeat_action = getattr(signals, "repeat_action_count", 0)
        repeat_read = getattr(signals, "repeat_read_count", 0)
        repeat_list = getattr(signals, "repeat_list_count", 0)
        if repeat_action >= self._streak_threshold:
            repeat_tool = str(getattr(signals, "repeat_action_tool", "") or "")
            repeat_key = str(getattr(signals, "repeat_action_key", "") or "")
            _log.info(
                "[behavior.gate] repeat action streak=%d tool=%s key=%s",
                repeat_action,
                repeat_tool,
                repeat_key,
            )
            if (
                action.decision == "act"
                and str(action.chosen_action_id or "") == repeat_tool
                and action_key_param(action.params) == repeat_key
            ):
                if repeat_tool != "task.workbench" and self._has_tool("task.workbench"):
                    return self._build_repetition_workbench_gate(
                        action=action,
                        repeat_tool=repeat_tool,
                        repeat_key=repeat_key,
                        repeat_count=repeat_action,
                        domain="runtime-loop",
                        intent="停止重复低增量动作并恢复问题解决闭环",
                        rationale=(
                            f"行为门控改道：{repeat_tool} {repeat_key or '（空参数）'} "
                            f"已连续重复 {repeat_action} 次，本轮先沉淀证据并切换策略。"
                        ),
                    )
                return self._build_wait_fallback(
                    action=action,
                    rationale=(
                        f"行为门控改道：{repeat_tool} {repeat_key or '（空参数）'} "
                        f"已连续重复 {repeat_action} 次，继续执行没有新增证据；等待下一轮重新组装上下文或切换策略。"
                    ),
                )
        elif repeat_read >= self._streak_threshold:
            repeat_path = str(getattr(signals, "repeat_read_path", "") or "")
            _log.info(
                "[behavior.gate] repeat read streak=%d path=%s",
                repeat_read,
                repeat_path,
            )
            if (
                action.decision == "act"
                and str(action.chosen_action_id or "") == "file.read"
                and action_key_param(action.params) == repeat_path
            ):
                if self._has_tool("task.workbench"):
                    return self._build_repeat_path_workbench_gate(
                        action=action,
                        kind="read",
                        repeat_path=repeat_path,
                        repeat_count=repeat_read,
                    )
                return self._build_wait_fallback(
                    action=action,
                    rationale=(
                        f"行为门控改道：{repeat_path} 已连续重复读取 {repeat_read} 次，"
                        "继续执行没有新增证据；等待下一轮重新组装上下文或切换策略。"
                    ),
                )
        elif repeat_list >= self._streak_threshold:
            repeat_path = str(getattr(signals, "repeat_list_path", "") or "")
            _log.info(
                "[behavior.gate] repeat list streak=%d path=%s",
                repeat_list,
                repeat_path,
            )
            if (
                action.decision == "act"
                and str(action.chosen_action_id or "") == "file.list"
                and action_key_param(action.params) == repeat_path
            ):
                if self._has_tool("task.workbench"):
                    return self._build_repeat_path_workbench_gate(
                        action=action,
                        kind="list",
                        repeat_path=repeat_path,
                        repeat_count=repeat_list,
                    )
                return self._build_wait_fallback(
                    action=action,
                    rationale=(
                        f"行为门控改道：{repeat_path} 已连续重复列表枚举 {repeat_list} 次，"
                        "继续执行没有新增证据；等待下一轮重新组装上下文或切换策略。"
                    ),
                )

        if self._should_gate_after_unprogressful_probe(action, signals):
            last_tool = str(getattr(signals, "last_action_tool", "") or "")
            last_key = str(getattr(signals, "last_action_key", "") or "")
            next_tool = str(action.chosen_action_id or "")
            next_key = action_key_param(action.params)
            reason = str(getattr(signals, "last_action_progress_reason", "") or "")
            if self._has_tool("task.workbench"):
                return self._build_unprogressful_probe_gate(
                    action=action,
                    last_tool=last_tool,
                    last_key=last_key,
                    next_tool=next_tool,
                    next_key=next_key,
                    reason=reason,
                )
            return self._build_wait_fallback(
                action=action,
                rationale=(
                    f"行为门控改道：上一动作 {last_tool} 未推进，本轮仍选择同类探索 {next_tool}；"
                    "缺少 task.workbench，先等待下一轮重新收敛。"
                ),
            )

        return action

    def _build_workbench_gate(
        self,
        *,
        action: JudgmentOutput,
        domain: str,
        intent: str,
        evidence: list[str],
        hypothesis: str,
        next_verification: str,
        completion_checks: list[str],
        rationale: str,
        next_step: str | None = None,
        recovery_state: str | None = None,
    ) -> JudgmentOutput:
        workbench = {
            "domain": domain,
            "intent": intent,
            "evidence": evidence,
            "hypothesis": hypothesis,
            "next_verification": next_verification,
            "completion_checks": completion_checks,
        }
        if recovery_state:
            workbench["recovery_state"] = recovery_state
        return JudgmentOutput(
            decision="act",
            chosen_action_id="task.workbench",
            params={"workbench": workbench},
            rationale=rationale,
            reflection=action.reflection,
            next_step=next_step,
            model_strategy=dict(action.model_strategy or {}),
            applied_skills=list(action.applied_skills or []),
        )

    def _build_wait_fallback(self, *, action: JudgmentOutput, rationale: str) -> JudgmentOutput:
        fallback_rationale = rationale
        if "行为门控制动" not in fallback_rationale:
            fallback_rationale = f"行为门控制动：{fallback_rationale}"
        return JudgmentOutput(
            decision="wait",
            rationale=fallback_rationale,
            reflection=action.reflection,
            next_step=action.next_step,
            model_strategy=dict(action.model_strategy or {}),
            applied_skills=list(action.applied_skills or []),
        )

    def _build_repetition_workbench_gate(
        self,
        *,
        action: JudgmentOutput,
        repeat_tool: str,
        repeat_key: str,
        repeat_count: int,
        domain: str,
        intent: str,
        rationale: str,
    ) -> JudgmentOutput:
        return self._build_workbench_gate(
            action=action,
            domain=domain,
            intent=intent,
            evidence=[
                f"{repeat_tool or 'unknown'} {repeat_key or '（空参数）'} 已连续重复 {repeat_count} 次。",
                "继续执行同一动作无法提供新增证据，应先综合已有结果或切换验证方式。",
            ],
            hypothesis="当前卡点不是缺少工具调用，而是缺少对重复结果的综合判断和策略切换。",
            next_verification=(
                f"不要再重复执行 {repeat_tool or '同一工具'} {repeat_key or ''}；"
                "先总结已有证据，或改用不同工具/参数验证同一假设。"
            ),
            completion_checks=[
                "已停止重复同一低增量动作。",
                "已把重复结果转化为结论、修正假设或新的验证动作。",
            ],
            rationale=rationale,
            next_step=(
                f"停止重复 {repeat_tool} {repeat_key or ''}，先综合已有证据；"
                "仍需验证时改用不同证据源。"
            ),
        )

    def _build_repeat_path_workbench_gate(
        self,
        *,
        action: JudgmentOutput,
        kind: str,
        repeat_path: str,
        repeat_count: int,
    ) -> JudgmentOutput:
        if kind == "list":
            return self._build_workbench_gate(
                action=action,
                domain="runtime-loop",
                intent="停止重复目录枚举并恢复问题解决闭环",
                evidence=[
                    f"file.list 已连续 {repeat_count} 次列出相同目录结果: {repeat_path}",
                    "继续列同一目录无法提供新增证据，应先基于已有目录结果选择具体文件、命令或结论。",
                ],
                hypothesis="当前循环不是缺少目录信息，而是缺少从目录清单到具体验证对象的收敛。",
                next_verification=(
                    f"不要再列出 {repeat_path}；从已有目录结果中选择最相关文件读取，"
                    "或改用 grep/测试/配置查询等更具体证据源。"
                ),
                completion_checks=[
                    "已停止重复枚举同一目录。",
                    "已把目录结果转化为具体文件读取、验证命令或可回答结论。",
                ],
                rationale=(
                    f"行为门控改道：{repeat_path} 已连续重复列表枚举 {repeat_count} 次，"
                    "本轮先沉淀证据并要求切换到具体验证对象。"
                ),
                next_step=(
                    f"停止重复列出 {repeat_path}，先选择具体文件/命令验证；"
                    "仍需目录信息时换更精确路径。"
                ),
            )
        return self._build_workbench_gate(
            action=action,
            domain="runtime-loop",
            intent="停止重复读取并恢复问题解决闭环",
            evidence=[
                f"file.read 已连续 {repeat_count} 次读取同一路径: {repeat_path}",
                "继续读取同一路径无法提供新增证据，应先总结已读内容或切换验证手段。",
            ],
            hypothesis="当前循环不是缺少读取能力，而是缺少读取后的证据综合和下一步改道。",
            next_verification=(
                f"不要再读取 {repeat_path}；基于已读结果写出判断，"
                "或改用 grep/测试/配置查询等不同证据源验证。"
            ),
            completion_checks=[
                "已停止重复读取同一路径。",
                "已把已有读取结果转化为结论或新的验证动作。",
            ],
            rationale=(
                f"行为门控改道：{repeat_path} 已连续重复读取 {repeat_count} 次，"
                "本轮先沉淀证据并要求切换验证策略。"
            ),
            next_step=(
                f"停止重复读取 {repeat_path}，先总结已读证据；"
                "仍需验证时改用不同证据源。"
            ),
        )

    def _build_unprogressful_probe_gate(
        self,
        *,
        action: JudgmentOutput,
        last_tool: str,
        last_key: str,
        next_tool: str,
        next_key: str,
        reason: str,
    ) -> JudgmentOutput:
        return self._build_workbench_gate(
            action=action,
            domain="runtime-loop",
            intent="上一低增量探索未推进，停止同类重复并收敛问题解决闭环",
            evidence=[
                f"上一动作 {last_tool or 'unknown'} {last_key or '（空参数）'} 被判定为未推进。",
                f"本轮又选择同类探索动作 {next_tool or 'unknown'} {next_key or '（空参数）'}。",
                f"未推进原因: {reason or '系统未给出额外原因'}",
            ],
            hypothesis="当前不是缺少更多列表/搜索，而是需要先综合已有证据，明确下一条高信息增量验证。",
            next_verification=(
                "不要继续同类 list/search 枚举；先总结已有结果，"
                "若仍需取证，改用更具体的文件读取、测试、配置查询或用户可见结论。"
            ),
            completion_checks=[
                "已停止低增量探索循环。",
                "已把未推进原因转化为新的验证策略或可回答结论。",
            ],
            rationale=(
                f"行为门控改道：上一动作 {last_tool} 未推进，"
                f"本轮仍选择同类探索 {next_tool}，先写 workbench 收敛。"
            ),
            next_step="先综合已有证据并选择更具体的高信息增量验证动作。",
        )

    def _should_gate_after_unprogressful_probe(self, action: JudgmentOutput, signals: Any) -> bool:
        if action.decision != "act":
            return False
        next_tool = str(action.chosen_action_id or "").strip()
        if not next_tool or next_tool == "task.workbench":
            return False
        if getattr(signals, "last_action_progressful", None) is not False:
            return False
        if str(getattr(signals, "last_action_status", "") or "") not in {"ok", "skipped"}:
            return False
        last_tool = str(getattr(signals, "last_action_tool", "") or "").strip()
        if not last_tool:
            return False
        if next_tool == last_tool and (
            next_tool in _LOW_INCREMENT_EXPLORATION_TOOLS
            or tool_has_capability(self._registry, next_tool, "completion_info_only")
            or tool_has_capability(self._registry, next_tool, "ask_evidence")
        ):
            return True
        return (
            last_tool in _LOW_INCREMENT_EXPLORATION_TOOLS
            and next_tool in _LOW_INCREMENT_EXPLORATION_TOOLS
            and _LOW_INCREMENT_TOOL_GROUPS.get(last_tool) == _LOW_INCREMENT_TOOL_GROUPS.get(next_tool)
        )

    def _should_gate_evidence_task_wait(self, action: JudgmentOutput, signals: Any) -> bool:
        if action.decision not in {"wait", "pause"}:
            return False
        if not self._has_tool("task.workbench"):
            return False
        source = str(getattr(signals, "active_task_source", "") or "").strip()
        if source != "self_drive":
            return False
        status = str(getattr(signals, "active_task_status", "") or "").strip()
        if status in {"done", "cancelled"}:
            return False
        next_step = str(getattr(signals, "active_task_next_step", "") or "").strip()
        if not next_step:
            return False
        lowered = next_step.lower()
        wait_streak = int(getattr(signals, "wait_streak", 0) or 0)
        has_evidence = any(marker in lowered for marker in _EVIDENCE_MARKERS)
        if status == "waiting" and not has_evidence:
            return False
        if any(marker in lowered for marker in _WAIT_MARKERS) and not has_evidence:
            return wait_streak >= 3
        return has_evidence

    def _has_tool(self, name: str) -> bool:
        if not name:
            return False
        registry = self._registry
        if registry is None:
            try:
                from tools.registry import default_tool_registry

                registry = default_tool_registry()
            except Exception:
                return False
        get = getattr(registry, "get", None)
        if not callable(get):
            return False
        return get(name) is not None

    def apply_cognitive_probe(self, signals: Any) -> None:
        """将当前循环探针状态写入 cognitive_signals 对象（原地修改）。"""
        signals.repeat_action_count = self._action_streak_count
        signals.repeat_action_tool = (self._action_streak_sig or ("", ""))[0]
        signals.repeat_action_key = (self._action_streak_sig or ("", ""))[1]
        signals.repeat_read_count = self._read_streak_count
        signals.repeat_read_path = (self._read_streak_fp or ("", 0, ""))[0]
        signals.repeat_list_count = self._list_streak_count
        signals.repeat_list_path = (self._list_streak_fp or ("", ""))[0]
        signals.loop_probe_version = self._loop_probe_version

    def snapshot(self) -> dict[str, Any]:
        return {
            "action_streak_sig": self._action_streak_sig,
            "action_streak_count": self._action_streak_count,
            "wait_streak": self._wait_streak,
        }

    # ── 追踪方法 ──────────────────────────────────────────────────────────────

    def on_act(
        self,
        tool_id: str,
        key_param: str,
        task_id: str | None,
        params: dict | None = None,
    ) -> list[WMItem]:
        """追踪 act 行为（file.read 和非 file.read 均需调用）。

        - action streak：非 file.read / file.list 工具连续相同时返回 WMItem
          对 file.edit / file.write 使用内容指纹尺化 key，避免同文件不同内容的导致假阳性。

        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        items: list[WMItem] = []

        if tool_has_capability(self._registry, tool_id, "result_streak_only"):
            return items  # file.read / file.list streak 由结果感知处理
        if not task_id:
            return items

        # 对编辑类工具，把内容指纹混入 key，避免同文件不同内容的连续误判为循环
        _effective_key = key_param
        if tool_has_capability(self._registry, tool_id, "completion_mutation") and params:
            _p = params or {}
            _content_sig = str(_p.get("old_text") or _p.get("content") or "")[:80]
            if _content_sig:
                _fp = hashlib.md5(_content_sig.encode("utf-8", errors="replace")).hexdigest()[:8]
                _effective_key = f"{key_param}#{_fp}"

        # action streak 检测（非 file.read）
        _sig = (tool_id, _effective_key)
        self._recent_actions.append(_sig)
        if self._action_streak_sig == _sig:
            self._action_streak_count += 1
        else:
            self._action_streak_sig = _sig
            self._action_streak_count = 1
        self._loop_probe_version += 1

        _n = self._streak_threshold
        if (
            len(self._recent_actions) == _n
            and len(set(self._recent_actions)) == 1
            and tool_id
        ):
            _log.info("[self-awareness] 连续 %d 次相同行为: %s %s", _n, tool_id, key_param)
            items.append(WMItem(
                kind="self_awareness",
                content=f"[行为信号] 过去 {_n} 次均执行了 ({tool_id}, {key_param or '相同参数'})。",
                priority=self._pri_behavior_loop,
            ))
        return items

    def on_act_result(self, tool_id: str, result_summary: str) -> None:
        """act 执行后修正 streak：如果本次结果与上次不同，说明有实质进展，将 streak 折回 1。

        未防止属于 file.read / file.list （它们由各自的 on_read / on_list 处理）。
        """
        if tool_has_capability(self._registry, tool_id, "result_streak_only"):
            return
        fp = hashlib.md5((result_summary or "").encode("utf-8", errors="replace")).hexdigest()[:12]
        if fp and fp != self._last_act_result_fp:
            # 结果发生变化 → 实际有进展，将 streak 计数折回 1
            self._action_streak_count = 1
        self._last_act_result_fp = fp

    def on_read(
        self,
        path: str,
        max_chars: int,
        result_summary: str,
        *,
        start: int = 0,
        end: int = 0,
    ) -> list[WMItem]:
        """追踪 file.read 重复读取。

        两层检测：
        1. 内容 MD5 去重 — 同一文件同一窗口读多次
        2. 同文件顺序窗口探测 — 同一文件被分窗口连续读取（不同窗口内容不同但模式重复）

        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        _body = result_summary.split("\n", 1)[1] if "\n" in result_summary else result_summary
        _digest = hashlib.md5(_body.encode("utf-8", errors="replace")).hexdigest()[:12]
        _fp = (path, max_chars, _digest)

        self._recent_read_fps.append(_fp)
        if self._read_streak_fp == _fp:
            self._read_streak_count += 1
        else:
            self._read_streak_fp = _fp
            self._read_streak_count = 1
        self._loop_probe_version += 1

        items: list[WMItem] = []

        # 层 1：同内容重复
        _n = self._streak_threshold
        if len(self._recent_read_fps) == _n and len(set(self._recent_read_fps)) == 1:
            _log.info("[self-awareness] 连续 %d 次读取相同内容: %s", _n, path)
            items.append(WMItem(
                kind="self_awareness",
                content=f"[行为信号] 过去 {_n} 次均读取了相同内容 ({path})，MD5 一致。",
                priority=self._pri_behavior_loop,
            ))

        # 层 3：同文件多次读取（不同窗口/内容，但 path 相同）
        # 覆盖"跳跃读取同一文件"场景——Layer 1/2 均无法捕获此模式
        if (
            len(self._recent_read_fps) == _n
            and all(fp[0] == path for fp in self._recent_read_fps)
            and len({fp[2] for fp in self._recent_read_fps}) > 1  # 内容确实不同，排除 Layer 1 已处理的情况
        ):
            _log.info("[self-awareness] 连续 %d 次读取同一文件不同窗口: %s", _n, path)
            items.append(WMItem(
                kind="self_awareness",
                content=(
                    f"[行为信号] 已连续 {_n} 次读取同一文件的不同片段 ({path})。"
                    "若尚未定位目标内容，请明确行号范围后精确读取，或改用 shell.run grep 搜索关键词，避免全文扫描。"
                ),
                priority=self._pri_behavior_loop,
            ))

        # 层 2：同文件顺序窗口探测（不同窗口但模式为连续扫描）
        if start >= 0 and end > start:
            window_size = end - start
            if path == self._seq_window_path:
                gap = abs(start - self._seq_window_last_end)
                gap_ratio = gap / max(window_size, 1)
                # 窗口连续（与上次 end 接近）→ 递增计数
                if gap_ratio <= self._seq_window_gap_ratio or gap <= max(window_size, 100):
                    self._seq_window_count += 1
                else:
                    # 跳到了不连续区域 → 重置
                    self._seq_window_count = 1
            else:
                # 换文件了 → 重置
                self._seq_window_path = path
                self._seq_window_count = 1
                self._seq_window_warned = False
            self._seq_window_last_end = end

            if (
                self._seq_window_count >= self._seq_window_warn_at
                and not self._seq_window_warned
            ):
                self._seq_window_warned = True
                _log.info(
                    "[self-awareness] 同文件连续 %d 次窗口探测: %s",
                    self._seq_window_count, path,
                )
                items.append(WMItem(
                    kind="self_awareness",
                    content=f"[行为信号] 已连续 {self._seq_window_count} 次按窗口分段读取 ({path})。",
                    priority=self._pri_edit_caution,
                ))

        return items

    def on_list(self, path: str, result_summary: str) -> list[WMItem]:
        """追踪 file.list 相同目录结果的重复枚举。

        只在“同一路径 + 同样列表结果”连续出现时报警；
        同一路径但目录内容发生变化，不视为无效重复。
        """
        from memory.working import WMItem

        digest = hashlib.md5(result_summary.encode("utf-8", errors="replace")).hexdigest()[:12]
        fp = (path, digest)

        self._recent_list_fps.append(fp)
        if self._list_streak_fp == fp:
            self._list_streak_count += 1
        else:
            self._list_streak_fp = fp
            self._list_streak_count = 1
        self._loop_probe_version += 1

        items: list[WMItem] = []
        _n = self._streak_threshold
        if len(self._recent_list_fps) == _n and len(set(self._recent_list_fps)) == 1:
            _log.info("[self-awareness] 连续 %d 次列出相同目录结果: %s", _n, path)
            items.append(WMItem(
                kind="self_awareness",
                content=f"[行为信号] 过去 {_n} 次均列出了相同目录结果 ({path})，结果指纹一致。",
                priority=self._pri_behavior_loop,
            ))
        return items

    def on_wait(self, decision: str, has_active_task: bool) -> list[WMItem]:
        """追踪连续 wait/pause 决策，向 WM 注入状态汇报。

        原则：只汇报事实，由 LLM 自主决定是否继续等待。不做硬阻断。
        阈值来自配置（wait_streak_notify），每个阈值首次触达时发出一条通知。
        act 后重置计数。
        """
        from memory.working import WMItem

        if decision not in ("wait", "pause"):
            self._wait_streak = 0
            self._wait_streak_warned = set()
            return []

        self._wait_streak += 1
        items: list[WMItem] = []

        # 找到本轮应触发的最高阈值（每个阈值只触发一次）
        for thresh in self._wait_notify_thresholds:
            if self._wait_streak >= thresh and thresh not in self._wait_streak_warned:
                self._wait_streak_warned.add(thresh)
                # 阶段数：当前是第几个阈值（影响 priority，越晚越高）
                _stage = self._wait_notify_thresholds.index(thresh)
                _max_stage = max(len(self._wait_notify_thresholds) - 1, 1)
                _priority = round(0.85 + 0.1 * (_stage / _max_stage), 3)
                _msg = (
                    f"[行为汇报] 已连续 {self._wait_streak} 轮 {decision}。"
                    f" 任务存在：{'是' if has_active_task else '否'}。"
                    f" 通知 {_stage + 1}/{len(self._wait_notify_thresholds)}。"
                )
                _log.info("[behavior] wait-streak=%d, thresh=%d, priority=%.3f",
                          self._wait_streak, thresh, _priority)
                items.append(WMItem(kind="self_awareness", content=_msg, priority=_priority))
                break  # 每轮最多触发一条通知
        return items

    def on_judgment(self, rationale: str) -> list[WMItem]:
        """追踪 LLM rationale 指纹，检测"信念固化"（连续相同结论）。

        将 rationale 前 belief_hash_prefix 字符规范化后计算 MD5 指纹。
        若同一指纹连续出现 >= belief_stale_threshold 次，注入 WM 警告。
        警告仅触发一次（_belief_stale_warned），直到结论真正改变后重置。

        返回需注入 WM 的条目列表（通常为空或 1 项）。
        """
        from memory.working import WMItem

        items: list[WMItem] = []
        if not rationale or not rationale.strip():
            return items

        # 规范化：去首尾空白、折叠空白、转小写
        normalized = " ".join(rationale.strip().split()).lower()
        fp = hashlib.md5(normalized.encode("utf-8", errors="replace")).hexdigest()[:12]

        self._rationale_hashes.append(fp)
        if self._belief_stale_hash == fp:
            self._belief_stale_count += 1
        else:
            # 结论改变 → 重置警告状态
            self._belief_stale_hash = fp
            self._belief_stale_count = 1
            self._belief_stale_warned = False

        if (
            self._belief_stale_count >= self._belief_stale_threshold
            and not self._belief_stale_warned
        ):
            self._belief_stale_warned = True
            _log.info(
                "[self-awareness] rationale 指纹连续 %d 次相同 (fp=%s)，可能存在信念固化",
                self._belief_stale_count, fp,
            )
            items.append(WMItem(
                kind="self_awareness",
                content=f"[认知信号] 推理结论已连续 {self._belief_stale_count} 轮基本相同（指纹 {fp}）。",
                priority=self._pri_belief_stale,
            ))
        return items

    def on_edit_failure(self, error: str) -> list:
        """追踪 file.edit 失败，连续失败时返回感知信号。

        LLM 可感知信号，自行决定是否切换策略（如用 file.write 或 shell.run sed 代替 file.edit）。
        """
        from memory.working import WMItem
        if "OldTextNotFound" in (error or ""):
            self._edit_fail_count += 1
            if self._edit_fail_count >= 2:
                return [WMItem(
                    kind="behavior_sense",
                    content=f"[感知] file.edit 已连续 {self._edit_fail_count} 次因 oldText 不匹配而失败。",
                    priority=self._pri_edit_caution,
                )]
        else:
            self._edit_fail_count = 0
        return []
