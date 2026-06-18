"""测试 Patch A（_ActionResultSummary 构建）、Patch B（一致性检测）、Patch C（WM 显著性门控）。

覆盖两个核心机制的行为正确性与边界情况。
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tool_result(
    *,
    summary: str = "",
    error: str | None = None,
    skipped: bool = False,
) -> Any:
    from tools.registry import ToolResult
    return ToolResult(summary=summary, error=error, skipped=skipped)


def _make_judgment_output(
    *,
    decision: str = "act",
    chosen_action_id: str = "file.edit",
    speech_intent: str = "",
) -> Any:
    from core.judgment.output import JudgmentOutput
    j = JudgmentOutput(decision=decision, chosen_action_id=chosen_action_id)
    j.speech_intent = speech_intent
    return j


def _make_history_entry(tool: str, status: str, result: str = "") -> dict[str, Any]:
    return {"tool": tool, "params": {}, "result": result, "status": status, "error": ""}


def _wm_item(kind: str, content: str, priority: float = 0.5):
    from memory.working import WMItem
    return WMItem(kind=kind, content=content, priority=priority)


# ─────────────────────────────────────────────────────────────────────────────
# Patch A：_ActionResultSummary 构建逻辑
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildActionResultSummary:
    """_build_action_result_summary 应正确从 tool_history / ToolResult 派生状态。"""

    def _build(self, action, result, tool_history):
        from core.loop.tick import _build_action_result_summary
        return _build_action_result_summary(action, result, tool_history)

    def test_act_ok_from_history(self):
        action = _make_judgment_output(decision="act", chosen_action_id="file.edit")
        result = _make_tool_result(summary="写入成功")
        history = [_make_history_entry("file.edit", "ok", "写入成功")]
        ar = self._build(action, result, history)
        assert ar.action_ran is True
        assert ar.action_succeeded is True
        assert ar.tool_name == "file.edit"
        assert "写入" in ar.summary

    def test_act_error_from_history(self):
        action = _make_judgment_output(decision="act", chosen_action_id="shell.run")
        result = _make_tool_result(summary="", error="Permission denied")
        history = [_make_history_entry("shell.run", "error", "")]
        ar = self._build(action, result, history)
        assert ar.action_ran is True
        assert ar.action_succeeded is False

    def test_act_skipped_from_history(self):
        action = _make_judgment_output(decision="act", chosen_action_id="file.read")
        result = _make_tool_result(skipped=True)
        history = [_make_history_entry("file.read", "skipped", "")]
        ar = self._build(action, result, history)
        assert ar.action_ran is True
        assert ar.action_succeeded is None  # skipped → 不确定

    def test_wait_decision_not_ran(self):
        action = _make_judgment_output(decision="wait")
        result = _make_tool_result()
        ar = self._build(action, result, [])
        assert ar.action_ran is False
        assert ar.action_succeeded is None

    def test_fallback_to_tool_result_when_history_empty(self):
        """history 为空时应回退到直接读 ToolResult。"""
        action = _make_judgment_output(decision="act", chosen_action_id="file.write")
        result = _make_tool_result(summary="内容已写入", error=None, skipped=False)
        ar = self._build(action, result, [])  # 空 history
        assert ar.action_ran is True
        assert ar.action_succeeded is True  # ToolResult 无 error 无 skipped → True

    def test_fallback_to_tool_result_error(self):
        action = _make_judgment_output(decision="act", chosen_action_id="exec.run")
        result = _make_tool_result(error="timeout", skipped=False)
        ar = self._build(action, result, [])
        assert ar.action_succeeded is False

    def test_summary_keeps_full_text(self):
        long_summary = "x" * 500
        action = _make_judgment_output(decision="act", chosen_action_id="file.read")
        result = _make_tool_result(summary=long_summary)
        history = [_make_history_entry("file.read", "ok", long_summary)]
        ar = self._build(action, result, history)
        assert ar.summary == long_summary

    def test_tool_name_from_history_takes_priority(self):
        """history 末尾的 tool 名优先于 action.chosen_action_id。"""
        action = _make_judgment_output(decision="act", chosen_action_id="file.read")
        result = _make_tool_result()
        # history 末尾显示实际执行的是另一个工具
        history = [
            _make_history_entry("file.read", "ok"),
            _make_history_entry("file.write", "ok"),
        ]
        ar = self._build(action, result, history)
        assert ar.tool_name == "file.write"


# ─────────────────────────────────────────────────────────────────────────────
# Patch B：_check_mouth_consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckMouthConsistency:
    """_check_mouth_consistency 应在承诺措辞矛盾时记录警告，不抛异常，不改文本。"""

    def _check(self, reply, action_ran, succeeded, tool_name="file.edit"):
        from core.loop.tick import _ActionResultSummary, _check_mouth_consistency
        ar = _ActionResultSummary(
            action_ran=action_ran,
            action_succeeded=succeeded,
            tool_name=tool_name,
            summary="",
            error="",
        )
        _check_mouth_consistency(reply, ar)  # 不应抛出

    def test_no_warning_when_not_ran(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="lingzhou.loop"):
            self._check("马上去做", action_ran=False, succeeded=None)
        # action_ran=False 时不应触发检测
        assert not any("mouth-check" in r.message for r in caplog.records)

    def test_no_warning_when_failed(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="lingzhou.loop"):
            self._check("马上去做", action_ran=True, succeeded=False)
        assert not any("mouth-check" in r.message for r in caplog.records)

    def test_no_warning_when_done_marker_present(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="lingzhou.loop"):
            self._check("已经完成了修改", action_ran=True, succeeded=True)
        assert not any("mouth-check" in r.message for r in caplog.records)

    def test_warning_when_premature_and_succeeded(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="lingzhou.loop"):
            self._check("马上去做这件事", action_ran=True, succeeded=True)
        assert any("mouth-check" in r.message for r in caplog.records)

    def test_warning_for_various_premature_words(self, caplog):
        import logging
        for phrase in ("立即执行", "现在去修改", "我将处理", "接下来我会完成"):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="lingzhou.loop"):
                self._check(f"好的，{phrase}", action_ran=True, succeeded=True)
            assert any("mouth-check" in r.message for r in caplog.records), \
                f"未触发警告: phrase='{phrase}'"

    def test_no_warning_when_succeeded_and_clean_done_reply(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="lingzhou.loop"):
            self._check("刚刚写好了，请查看 README.md", action_ran=True, succeeded=True)
        assert not any("mouth-check" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Patch C：WorkingMemory.salience_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestSalienceGate:
    """salience_gate 应正确 boost 相关条目、丢弃低优无关条目、无条件保留 preserve_kinds。"""

    def _wm(self):
        from memory.working import WorkingMemory
        return WorkingMemory(capacity=30)

    def test_preserve_kinds_always_kept(self):
        from memory.working import TASK_SWITCH_PRESERVE_KINDS

        wm = self._wm()
        for kind in TASK_SWITCH_PRESERVE_KINDS:
            wm.add(_wm_item(kind, f"{kind} 上下文", priority=0.1))
        wm.add(_wm_item("noise", "无关内容 abc", priority=0.1))

        dropped = wm.salience_gate(
            "用户消息",
            preserve_kinds=set(TASK_SWITCH_PRESERVE_KINDS),
            priority_floor=0.7,
        )
        kinds = {item["kind"] for item in wm.get_top()}
        assert TASK_SWITCH_PRESERVE_KINDS.issubset(kinds)
        assert dropped >= 1  # noise 应被丢弃

    def test_high_priority_items_kept(self):
        wm = self._wm()
        wm.add(_wm_item("scheduler", "调度信号", priority=0.85))
        wm.add(_wm_item("low_noise", "无关低优 xyz", priority=0.3))

        wm.salience_gate("任意消息", preserve_kinds=set(), priority_floor=0.7)
        kinds = {item["kind"] for item in wm.get_top()}
        assert "scheduler" in kinds
        assert "low_noise" not in kinds

    def test_keyword_overlap_boosts_and_keeps(self):
        wm = self._wm()
        # 低优但内容与用户消息相关
        wm.add(_wm_item("task_progress", "README 英文版进度已完成 80%", priority=0.4))
        wm.add(_wm_item("irrelevant", "postgres 数据库连接问题", priority=0.4))

        wm.salience_gate(
            "英文版 README 怎么样了",
            preserve_kinds=set(),
            priority_floor=0.7,
            keyword_boost=0.2,
        )
        items = wm.get_top()
        kinds_map = {item["kind"]: item["priority"] for item in items}

        # README 相关条目应被保留且优先级提升
        assert "task_progress" in kinds_map
        assert kinds_map["task_progress"] > 0.4

        # 无关低优条目应被丢弃
        assert "irrelevant" not in kinds_map

    def test_boost_capped_at_1(self):
        wm = self._wm()
        wm.add(_wm_item("task_progress", "README 英文版", priority=0.95))
        wm.salience_gate("英文版 README", preserve_kinds=set(), priority_floor=0.7, keyword_boost=0.2)
        items = wm.get_top()
        for item in items:
            assert item["priority"] <= 1.0

    def test_empty_message_no_boost(self):
        """空消息时不做 boost，只按 priority_floor 过滤。"""
        wm = self._wm()
        wm.add(_wm_item("task_progress", "README 进度", priority=0.4))
        wm.add(_wm_item("high", "高优信号", priority=0.9))

        dropped = wm.salience_gate("", preserve_kinds=set(), priority_floor=0.7)
        items = wm.get_top()
        kinds = {item["kind"] for item in items}
        assert "high" in kinds
        assert "task_progress" not in kinds
        assert dropped == 1

    def test_returns_dropped_count(self):
        wm = self._wm()
        # 内容与用户消息（"今天天气怎么样"）毫无重叠
        for i in range(5):
            wm.add(_wm_item(f"dbcfg_{i}", f"数据库连接池配置参数{i}", priority=0.1))
        wm.add(_wm_item("anchor", "锚点", priority=0.9))

        dropped = wm.salience_gate("今天天气怎么样", preserve_kinds=set(), priority_floor=0.7)
        assert dropped == 5

    def test_no_items_dropped_when_all_high_priority(self):
        wm = self._wm()
        for i in range(3):
            wm.add(_wm_item(f"sig_{i}", f"高优信号{i}", priority=0.8))
        dropped = wm.salience_gate("任意消息", preserve_kinds=set(), priority_floor=0.7)
        assert dropped == 0
        assert len(wm) == 3

    def test_heap_integrity_after_gate(self):
        """gate 之后 WM 堆结构完好，add/get_top 不应崩溃。"""
        from memory.working import WMItem
        wm = self._wm()
        wm.add(_wm_item("a", "内容 A README", priority=0.4))
        wm.add(_wm_item("b", "内容 B 无关", priority=0.3))
        wm.add(_wm_item("c", "高优信号 C", priority=0.9))

        wm.salience_gate("README 进度", preserve_kinds=set(), priority_floor=0.7, keyword_boost=0.2)

        # 之后继续 add 不崩
        wm.add(WMItem(kind="new", content="新条目", priority=0.85))
        top = wm.get_top()
        assert any(item["kind"] == "new" for item in top)


# ─────────────────────────────────────────────────────────────────────────────
# _wm_keywords / _has_wm_overlap helper 单元测试
# ─────────────────────────────────────────────────────────────────────────────

class TestWmKeywords:
    def test_cjk_ngram(self):
        from memory.working import _wm_keywords
        kws = _wm_keywords("英文版README进度")
        # 应包含 2-gram
        assert "英文" in kws
        assert "文版" in kws
        assert "进度" in kws

    def test_ascii_words(self):
        from memory.working import _wm_keywords
        kws = _wm_keywords("check README status")
        assert "readme" in kws
        assert "status" in kws
        # 停用词 "the" 不应出现
        assert "the" not in kws

    def test_short_ascii_filtered(self):
        from memory.working import _wm_keywords
        kws = _wm_keywords("is it ok now")
        # 长度 < 3 的词不入集合
        assert "is" not in kws
        assert "it" not in kws
        assert "ok" not in kws  # 长度 2

    def test_has_overlap_true(self):
        from memory.working import _has_wm_overlap, _wm_keywords
        kws = _wm_keywords("英文版 README 怎么样了")
        assert _has_wm_overlap("README 英文版进度已完成 80%", kws)

    def test_has_overlap_false(self):
        from memory.working import _has_wm_overlap, _wm_keywords
        kws = _wm_keywords("英文版 README 怎么样了")
        assert not _has_wm_overlap("postgres 数据库连接超时", kws)

    def test_empty_message(self):
        from memory.working import _wm_keywords
        assert _wm_keywords("") == frozenset()
