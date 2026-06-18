"""全方位测试 口腔器官（oral organ） 与 大脑皮层（cortex/WM salience_gate）。

聚焦两条链路：
  口腔器官链路：
    _build_action_result_summary
    → _ActionResultSummary 注入 _build_continue_context
    → assembler 产出正确字符串
    → _finalize_tick_user_reply 正确路由（user_message 有/无）
    → 字段正确传播（rationale/reflection/next_step）
    → 兜底 fallback 行为

  大脑皮层/WM 链路：
    salience_gate 在不同 priority / keyword 情况下的保留/丢弃/boost 行为
    _wm_keywords / _has_wm_overlap 的极端情况
    speech_intent 降级（大脑产出意图草稿，不直接对外发话）
    WM 容量压力下 gate 后堆完整性
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 公共 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ar(
    *,
    action_ran: bool = True,
    action_succeeded: bool | None = True,
    tool_name: str = "file.edit",
    summary: str = "写入成功",
    error: str = "",
) -> Any:
    from core.loop.tick import _ActionResultSummary
    return _ActionResultSummary(
        action_ran=action_ran,
        action_succeeded=action_succeeded,
        tool_name=tool_name,
        summary=summary,
        error=error,
    )


def _make_emotion(
    *,
    dominant: str = "satisfied",
    valence: float = 0.72,
    arousal: float = 0.35,
    strategy: str = "maintain",
) -> dict[str, Any]:
    return {"dominant": dominant, "valence": valence, "arousal": arousal,
            "regulation_strategy": strategy}


def _bare_assembler(prev_context: str = "PREV_CONTEXT") -> Any:
    """绕过 __init__ 创建最小 assembler 实例，仅用于测试 _build_continue_context。"""
    from core.judgment.assembler import JudgmentContextAssembler
    asm = object.__new__(JudgmentContextAssembler)
    asm._last_context_text = prev_context
    return asm


def _build_ctx(
    *,
    tool_history: list[dict] | None = None,
    reply_only: bool = True,
    user_message: str = "请帮我修改文件",
    wm_delta: list[dict] | None = None,
    speech_intent: str = "",
    action_result: Any = None,
    emotion_state: dict | None = None,
    prev_context: str = "PREV",
) -> str:
    asm = _bare_assembler(prev_context)
    return asm._build_continue_context(
        tool_history or [],
        user_message=user_message,
        reply_only=reply_only,
        wm_delta=wm_delta,
        speech_intent=speech_intent,
        action_result=action_result,
        emotion_state=emotion_state,
    )


def _make_judgment_output(
    *,
    decision: str = "pause",
    reply_to_user: str = "",
    speech_intent: str = "",
    rationale: str = "",
    reflection: str = "",
    next_step: str = "",
) -> Any:
    from core.judgment.output import JudgmentOutput
    j = JudgmentOutput(decision=decision, chosen_action_id="")
    j.reply_to_user = reply_to_user
    j.speech_intent = speech_intent
    j.rationale = rationale
    j.reflection = reflection
    j.next_step = next_step
    return j


def _make_tool_result(
    *,
    summary: str = "",
    error: str | None = None,
    skipped: bool = False,
    state_delta: dict[str, Any] | None = None,
) -> Any:
    from tools.registry import ToolResult
    return ToolResult(summary=summary, error=error, skipped=skipped, state_delta=state_delta or {})


def _fake_loop(
    *,
    decide_reply: str = "文件已成功写入。",
    decide_rationale: str = "",
    decide_reflection: str = "",
    decide_next_step: str = "",
) -> Any:
    """最小 mock loop，支持 _finalize_tick_user_reply 完整链路调用。"""
    async def _decide_continue(*args, **kwargs):
        j = _make_judgment_output(
            reply_to_user=decide_reply,
            rationale=decide_rationale,
            reflection=decide_reflection,
            next_step=decide_next_step,
        )
        return j

    regulation = SimpleNamespace(strategy="maintain")
    emotion = SimpleNamespace(dominant="satisfied", valence=0.72, arousal=0.35, regulation=regulation)

    loop_cfg = SimpleNamespace(
        chat_thinking=None,
        autonomous_thinking=None,
        workspace_dir="/tmp",
    )
    thresholds = SimpleNamespace(wm_pri_signal=0.7)
    cfg = SimpleNamespace(thinking=None, thresholds=thresholds, loop=loop_cfg)

    judgment = SimpleNamespace(decide_continue=_decide_continue)

    async def _get_fact(key: str):
        return (None, False)

    task_store = SimpleNamespace(
        get_fact=_get_fact,
        add_chat_message=AsyncMock(),
        get_recent_chat_messages=AsyncMock(return_value=[]),
    )

    return SimpleNamespace(
        _cfg=cfg,
        _emotion=emotion,
        _judgment=judgment,
        _pending_routing_overrides=None,
        _task_store=task_store,
        _episodic=None,
    )


def _wm_item(kind: str, content: str, priority: float = 0.5) -> Any:
    from memory.working import WMItem
    return WMItem(kind=kind, content=content, priority=priority)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1：assembler._build_continue_context — 口腔器官前语言消息块
# ─────────────────────────────────────────────────────────────────────────────

class TestAssemblerReplyOnlyBlocks:
    """_build_continue_context(reply_only=True) 应正确拼入执行状态与情绪块。"""

    def test_execution_block_present_when_action_result(self):
        ctx = _build_ctx(action_result=_make_ar())
        assert "## 本轮执行状态" in ctx

    def test_execution_succeeded_text(self):
        ctx = _build_ctx(action_result=_make_ar(action_succeeded=True))
        assert "成功" in ctx

    def test_execution_failed_text_and_error(self):
        ar = _make_ar(action_succeeded=False, error="Permission denied", summary="")
        ctx = _build_ctx(action_result=ar)
        assert "失败" in ctx
        assert "Permission denied" in ctx

    def test_execution_not_ran_text(self):
        ar = _make_ar(action_ran=False, action_succeeded=None, tool_name="", summary="")
        ctx = _build_ctx(action_result=ar)
        assert "未执行" in ctx

    def test_execution_skipped_text(self):
        ar = _make_ar(action_succeeded=None, summary="")
        ctx = _build_ctx(action_result=ar)
        assert "跳过" in ctx or "不确定" in ctx

    def test_tool_name_in_block(self):
        ctx = _build_ctx(action_result=_make_ar(tool_name="shell.run"))
        assert "shell.run" in ctx

    def test_summary_in_block(self):
        ctx = _build_ctx(action_result=_make_ar(summary="共写入 42 行"))
        assert "42 行" in ctx

    def test_emotion_block_present_when_emotion_state(self):
        ctx = _build_ctx(emotion_state=_make_emotion())
        assert "## 当前情绪状态" in ctx

    def test_emotion_dominant_in_block(self):
        ctx = _build_ctx(emotion_state=_make_emotion(dominant="curious"))
        assert "curious" in ctx

    def test_emotion_valence_arousal_in_block(self):
        ctx = _build_ctx(emotion_state=_make_emotion(valence=0.65, arousal=0.28))
        assert "0.65" in ctx
        assert "0.28" in ctx

    def test_emotion_strategy_in_block(self):
        ctx = _build_ctx(emotion_state=_make_emotion(strategy="downregulate"))
        assert "downregulate" in ctx

    def test_no_execution_block_when_no_action_result(self):
        ctx = _build_ctx(action_result=None)
        assert "## 本轮执行状态" not in ctx

    def test_no_emotion_block_when_no_emotion_state(self):
        ctx = _build_ctx(emotion_state=None)
        assert "## 当前情绪状态" not in ctx

    def test_speech_intent_hint_when_provided(self):
        ctx = _build_ctx(speech_intent="我打算修改 README")
        assert "README" in ctx

    def test_no_speech_intent_hint_when_empty(self):
        ctx = _build_ctx(speech_intent="")
        # 不应出现 intent 关键词
        assert "意图草稿" not in ctx

    def test_tool_calling_forbidden_hint(self):
        """reply_only 模式必须禁止工具调用。"""
        ctx = _build_ctx(reply_only=True)
        assert "禁止再调用任何工具" in ctx

    def test_prev_context_always_prepended(self):
        ctx = _build_ctx(prev_context="SENTINEL_PREV")
        assert ctx.startswith("SENTINEL_PREV")

    def test_both_blocks_appear_together(self):
        ctx = _build_ctx(
            action_result=_make_ar(action_succeeded=True),
            emotion_state=_make_emotion(),
        )
        assert "## 本轮执行状态" in ctx
        assert "## 当前情绪状态" in ctx
        # 执行状态在情绪状态之前
        assert ctx.index("## 本轮执行状态") < ctx.index("## 当前情绪状态")

    def test_continue_mode_no_execution_emotion_blocks(self):
        """reply_only=False（续判）不应出现口腔专属块。"""
        ctx = _build_ctx(
            reply_only=False,
            action_result=_make_ar(),
            emotion_state=_make_emotion(),
        )
        assert "## 本轮执行状态" not in ctx
        assert "## 当前情绪状态" not in ctx

    def test_wm_delta_block_appears(self):
        delta = [{"kind": "task_progress", "priority": 0.8, "content": "进度 80%"}]
        ctx = _build_ctx(wm_delta=delta, reply_only=False)
        assert "本轮新增工作记忆" in ctx
        assert "task_progress" in ctx

    def test_wm_delta_absent_when_none(self):
        ctx = _build_ctx(wm_delta=None, reply_only=False)
        assert "本轮新增工作记忆" not in ctx


# ─────────────────────────────────────────────────────────────────────────────
# Section 2：_finalize_tick_user_reply 路由逻辑
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalizeTickUserReplyRouting:
    """_finalize_tick_user_reply 应正确分流：user_message→口腔器官; 无消息→speech_intent。"""

    @pytest.mark.asyncio
    async def test_user_message_clears_draft_and_uses_oral_organ(self):
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="文件已成功写入。")
        action = _make_judgment_output(decision="act")
        action.reply_to_user = "旧草稿不应出现"
        action.speech_intent = "打算修改文件"
        result = _make_tool_result(summary="写入成功")

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "帮我修改文件", None, None
            )

        assert action.reply_to_user == "文件已成功写入。"
        # 旧草稿彻底清除，不出现
        assert "旧草稿" not in action.reply_to_user

    @pytest.mark.asyncio
    async def test_no_user_message_uses_speech_intent_directly(self):
        """自主 tick（无用户消息）：speech_intent 直接成为 reply_to_user，不走口腔器官。"""
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="口腔器官不应被调用")
        action = _make_judgment_output(decision="wait")
        action.speech_intent = "我有一个发现想分享"
        action.reply_to_user = ""
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "", None, None
            )

        assert action.reply_to_user == "我有一个发现想分享"

    @pytest.mark.asyncio
    async def test_no_message_no_intent_reply_stays_empty(self):
        """无用户消息 + 无 speech_intent → reply_to_user 保持为空。"""
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="不应出现")
        action = _make_judgment_output(decision="wait")
        action.speech_intent = ""
        action.reply_to_user = ""
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "", None, None
            )

        assert action.reply_to_user == ""

    @pytest.mark.asyncio
    async def test_oral_organ_empty_reply_triggers_fallback(self):
        """口腔器官未生成回复时应触发兜底 fallback，而非空字符串。"""
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="")   # 口腔器官返回空
        action = _make_judgment_output(decision="wait", rationale="等待信息")
        action.reply_to_user = ""
        action.speech_intent = ""
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "用户消息", None, None
            )

        # fallback 一定非空
        assert action.reply_to_user != ""

    @pytest.mark.asyncio
    async def test_fallback_contains_error_info_on_tool_failure(self):
        """工具失败时 fallback 应包含错误语义。"""
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="")   # 口腔器官返回空
        action = _make_judgment_output(decision="act")
        action.speech_intent = ""
        action.reply_to_user = ""
        result = _make_tool_result(error="file not found")

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "帮我读取文件", None, None
            )

        assert "error" in action.reply_to_user.lower() or "状态" in action.reply_to_user


# ─────────────────────────────────────────────────────────────────────────────
# Section 3：口腔器官字段传播
# ─────────────────────────────────────────────────────────────────────────────

class TestOralOrganFieldPropagation:
    """口腔器官回复的 rationale/reflection/next_step 应正确传播到 action。"""

    @pytest.mark.asyncio
    async def test_rationale_propagated(self):
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="好的。", decide_rationale="文件写入成功，用户已得到回复")
        action = _make_judgment_output(decision="act")
        action.speech_intent = ""
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "请修改", None, None
            )

        assert "写入成功" in action.rationale

    @pytest.mark.asyncio
    async def test_reflection_propagated_when_action_had_none(self):
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="完成了。", decide_reflection="文件逻辑正确")
        action = _make_judgment_output(decision="act")
        action.reflection = ""  # 原本为空
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "请修改", None, None
            )

        assert action.reflection == "文件逻辑正确"

    @pytest.mark.asyncio
    async def test_reflection_not_overwritten(self):
        """action 已有 reflection 时，口腔器官的 reflection 不应覆盖。"""
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="完成了。", decide_reflection="口腔器官的新反思")
        action = _make_judgment_output(decision="act")
        action.reflection = "大脑皮层已有反思"  # 原本非空
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "请修改", None, None
            )

        assert action.reflection == "大脑皮层已有反思"

    @pytest.mark.asyncio
    async def test_next_step_propagated_when_action_had_none(self):
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="写完了。", decide_next_step="等待用户确认")
        action = _make_judgment_output(decision="act")
        action.next_step = ""
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "请修改", None, None
            )

        assert action.next_step == "等待用户确认"

    @pytest.mark.asyncio
    async def test_next_step_not_overwritten(self):
        from core.loop.tick import _finalize_tick_user_reply
        loop = _fake_loop(decide_reply="写完了。", decide_next_step="口腔器官的 next_step")
        action = _make_judgment_output(decision="act")
        action.next_step = "大脑皮层的 next_step"  # 原本非空
        result = _make_tool_result()

        with patch("core.loop.tick._persist_tick_user_reply", new=AsyncMock()):
            await _finalize_tick_user_reply(
                loop, action, result, [], "请修改", None, None
            )

        assert action.next_step == "大脑皮层的 next_step"


# ─────────────────────────────────────────────────────────────────────────────
# Section 4：speech_intent 降级逻辑（大脑皮层不直接对外发话）
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeechIntentDemotion:
    """JudgmentPhase 输出的 reply_to_user 仅在 act 时降级为 speech_intent。"""

    def test_reply_demoted_to_speech_intent(self):
        from core.judgment.output import JudgmentOutput
        action = JudgmentOutput(decision="act", chosen_action_id="file.edit")
        action.reply_to_user = "我马上帮你修改"
        action.speech_intent = ""

        # 模拟 _TickJudgmentPhase.run 中的降级逻辑
        if action.decision == "act" and action.reply_to_user:
            action.speech_intent = action.reply_to_user
            action.reply_to_user = ""

        assert action.speech_intent == "我马上帮你修改"
        assert action.reply_to_user == ""

    def test_no_demotion_when_reply_empty(self):
        from core.judgment.output import JudgmentOutput
        action = JudgmentOutput(decision="act", chosen_action_id="file.edit")
        action.reply_to_user = ""
        action.speech_intent = ""

        if action.decision == "act" and action.reply_to_user:
            action.speech_intent = action.reply_to_user
            action.reply_to_user = ""

        assert action.speech_intent == ""
        assert action.reply_to_user == ""

    def test_wait_reply_is_not_demoted(self):
        from core.judgment.output import JudgmentOutput

        action = JudgmentOutput(decision="wait", chosen_action_id="")
        action.reply_to_user = "这是最终答复"
        action.speech_intent = ""

        if action.decision == "act" and action.reply_to_user:
            action.speech_intent = action.reply_to_user
            action.reply_to_user = ""

        assert action.speech_intent == ""
        assert action.reply_to_user == "这是最终答复"

    def test_existing_speech_intent_overwritten_on_demotion(self):
        """若大脑已有旧 speech_intent，reply_to_user 降级应覆盖它。"""
        from core.judgment.output import JudgmentOutput
        action = JudgmentOutput(decision="act", chosen_action_id="file.edit")
        action.reply_to_user = "新的回复意图"
        action.speech_intent = "旧的意图"

        if action.decision == "act" and action.reply_to_user:
            action.speech_intent = action.reply_to_user
            action.reply_to_user = ""

        assert action.speech_intent == "新的回复意图"


class TestContinuePhaseAndAutonomousReplyGuard:
    @pytest.mark.asyncio
    async def test_continue_phase_demotes_reply_to_speech_intent(self):
        from core.loop.shared.continue_phase import _run_continue_phase

        cont = _make_judgment_output(decision="act", reply_to_user="我继续处理这张图")
        cont.chosen_action_id = "image.analyze"

        loop = SimpleNamespace(
            _cfg=SimpleNamespace(
                thinking=None,
                loop=SimpleNamespace(chat_thinking=None, autonomous_thinking=None),
                thresholds=SimpleNamespace(
                    continue_tool_history_compact_threshold=8,
                    continue_tool_history_keep_last=4,
                    wm_pri_insight=0.8,
                )
            ),
            _emotion=SimpleNamespace(valence=0.1, arousal=0.2),
            _task_store=SimpleNamespace(has_pending_chat_message=AsyncMock(return_value=False)),
            _judgment=SimpleNamespace(decide_continue=AsyncMock(return_value=cont)),
            _behavior=SimpleNamespace(
                on_act=lambda *args, **kwargs: [],
                apply_cognitive_probe=lambda *args, **kwargs: None,
                on_act_result=lambda *args, **kwargs: None,
                on_edit_failure=lambda *args, **kwargs: [],
            ),
            _execution=SimpleNamespace(dispatch=AsyncMock(return_value=_make_tool_result(summary="分析完成"))),
            _wm=SimpleNamespace(add=lambda *args, **kwargs: None),
            _episodic=SimpleNamespace(record=lambda *args, **kwargs: None),
            _registry=MagicMock(),
            _pending_routing_overrides=None,
        )

        with patch("core.loop.shared.continue_phase._maybe_reconcile_bootstrap", new=AsyncMock()):
            final_action, final_result = await _run_continue_phase(
                loop=loop,
                ctx=MagicMock(),
                user_message="请看图",
                active_task=SimpleNamespace(id=42),
                cognitive_signals=MagicMock(),
                action=_make_judgment_output(decision="act"),
                result=_make_tool_result(),
                tool_history=[],
            )

        assert final_action.speech_intent == "我继续处理这张图"
        assert final_action.reply_to_user == ""
        assert final_result.summary == "分析完成"

    @pytest.mark.asyncio
    async def test_continue_phase_stops_after_user_prompted_mutation(self):
        from conftest import _tool_registry

        from core.loop.shared.continue_phase import _run_continue_phase

        first = _make_judgment_output(decision="act")
        first.chosen_action_id = "file.write"
        first.params = {"path": "/tmp/a.txt", "content": "ok"}
        second = _make_judgment_output(decision="act")
        second.chosen_action_id = "file.read"
        second.params = {"path": "/tmp/a.txt"}

        decide_continue = AsyncMock(side_effect=[first, second])
        dispatch = AsyncMock(return_value=_make_tool_result(summary="写入完成"))

        loop = SimpleNamespace(
            _cfg=SimpleNamespace(
                thinking=None,
                loop=SimpleNamespace(chat_thinking=None, autonomous_thinking=None),
                thresholds=SimpleNamespace(
                    continue_tool_history_compact_threshold=8,
                    continue_tool_history_keep_last=4,
                    continue_max_inner_rounds=4,
                    wm_pri_insight=0.8,
                )
            ),
            _emotion=SimpleNamespace(valence=0.1, arousal=0.2),
            _task_store=SimpleNamespace(has_pending_chat_message=AsyncMock(return_value=False)),
            _judgment=SimpleNamespace(decide_continue=decide_continue),
            _behavior=SimpleNamespace(
                on_act=lambda *args, **kwargs: [],
                apply_cognitive_probe=lambda *args, **kwargs: None,
                apply_execution_gate=lambda action, signals: action,
                on_act_result=lambda *args, **kwargs: None,
                on_edit_failure=lambda *args, **kwargs: [],
            ),
            _execution=SimpleNamespace(dispatch=dispatch),
            _wm=SimpleNamespace(add=lambda *args, **kwargs: None),
            _episodic=SimpleNamespace(record=lambda *args, **kwargs: None),
            _registry=_tool_registry(),
            _pending_routing_overrides=None,
            _bootstrap_mode="none",
        )

        with patch("core.loop.shared.continue_phase._maybe_reconcile_bootstrap", new=AsyncMock()):
            final_action, final_result = await _run_continue_phase(
                loop=loop,
                ctx=MagicMock(),
                user_message="请修改这个文件",
                active_task=SimpleNamespace(id=42),
                cognitive_signals=MagicMock(),
                action=_make_judgment_output(decision="act"),
                result=_make_tool_result(),
                tool_history=[],
            )

        assert final_action.chosen_action_id == "file.write"
        assert final_result.summary == "写入完成"
        assert decide_continue.await_count == 1
        assert dispatch.await_count == 1

    @pytest.mark.asyncio
    async def test_duplicate_autonomous_reply_is_suppressed(self):
        from core.loop.tick import _persist_tick_user_reply

        task_store = SimpleNamespace(
            add_chat_message=AsyncMock(),
            get_recent_chat_messages=AsyncMock(
                return_value=[
                    {"id": 1, "role": "assistant", "content": "图片分析工具暂时报错。", "created_at": "2026-05-29 10:41:24"}
                ]
            ),
        )
        loop = SimpleNamespace(
            _task_store=task_store,
            _episodic=None,
            _emotion=SimpleNamespace(valence=0.0, arousal=0.0),
        )
        action = _make_judgment_output(decision="wait", reply_to_user="图片分析工具暂时报错。")

        with patch("core.loop.tick._resolve_reply_chat_id", new=AsyncMock(return_value="wechat:user-1")):
            await _persist_tick_user_reply(
                loop,
                action,
                SimpleNamespace(id=1127),
                "wechat:user-1",
                "",
            )

        task_store.add_chat_message.assert_not_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5：WM salience_gate 进阶场景
# ─────────────────────────────────────────────────────────────────────────────

class TestSalienceGateEdgeCases:
    """WM.salience_gate 在复杂组合下的行为。"""

    def _wm(self, capacity: int = 30) -> Any:
        from memory.working import WorkingMemory
        return WorkingMemory(capacity=capacity)

    def test_task_switch_preserve_kinds_are_kept_by_salience_gate(self):
        from memory.working import TASK_SWITCH_PRESERVE_KINDS, WorkingMemory

        wm = WorkingMemory(capacity=20)
        for kind in TASK_SWITCH_PRESERVE_KINDS:
            wm.add(_wm_item(kind, f"{kind} 上下文", priority=0.1))
        wm.add(_wm_item("noise", "无关垃圾项", priority=0.1))

        dropped = wm.salience_gate(
            "切换任务",
            preserve_kinds=set(TASK_SWITCH_PRESERVE_KINDS),
            priority_floor=0.72,
        )
        kinds = {i["kind"] for i in wm.get_top()}
        assert TASK_SWITCH_PRESERVE_KINDS.issubset(kinds)
        assert dropped >= 1

    def test_preserve_kinds_overrides_priority_floor(self):
        """priority=0 的 preserve_kinds 条目在任何情况下都不被丢弃。"""
        from memory.working import TASK_SWITCH_PRESERVE_KINDS

        wm = self._wm()
        wm.add(_wm_item("bootstrap_identity", "核心身份", priority=0.0))
        wm.add(_wm_item("task_anchor", "任务锚点", priority=0.0))
        wm.add(_wm_item("junk", "无用内容xyz", priority=0.0))

        dropped = wm.salience_gate(
            "今天天气怎么样",
            preserve_kinds=set(TASK_SWITCH_PRESERVE_KINDS),
            priority_floor=0.9,  # 极高 floor
        )
        kinds = {i["kind"] for i in wm.get_top()}
        assert "bootstrap_identity" in kinds
        assert "task_anchor" in kinds
        assert dropped >= 1

    def test_multiple_keywords_only_one_needs_to_match(self):
        """多关键词只需一个命中即可 boost。"""
        wm = self._wm()
        # 只含"README"，不含"英文"、"进度"
        wm.add(_wm_item("task_progress", "README 文件需要更新", priority=0.3))

        wm.salience_gate(
            "英文版 README 进度怎么样",
            preserve_kinds=set(),
            priority_floor=0.7,
            keyword_boost=0.2,
        )
        items = wm.get_top()
        kinds = {i["kind"] for i in items}
        assert "task_progress" in kinds  # README 命中，应保留

    def test_boost_raises_priority_above_floor(self):
        """boost 后 priority 应高于 priority_floor，确保后续不会被意外丢弃。"""
        from memory.working import WorkingMemory
        wm = WorkingMemory(capacity=30)
        wm.add(_wm_item("task_progress", "README 更新进度", priority=0.45))

        wm.salience_gate(
            "README 怎么样了",
            preserve_kinds=set(),
            priority_floor=0.6,
            keyword_boost=0.3,
        )
        items = wm.get_top()
        assert len(items) == 1
        assert items[0]["priority"] >= 0.6  # 0.45 + 0.3 = 0.75 > 0.6

    def test_gate_idempotent_on_second_call(self):
        """同一条件调用两次，第二次 dropped=0（已经过滤完毕）。"""
        wm = self._wm()
        wm.add(_wm_item("high", "高优条目", priority=0.9))
        wm.add(_wm_item("low", "低优无关内容xyz", priority=0.1))

        wm.salience_gate("今天天气", preserve_kinds=set(), priority_floor=0.7)
        dropped2 = wm.salience_gate("今天天气", preserve_kinds=set(), priority_floor=0.7)
        assert dropped2 == 0

    def test_small_capacity_wm_survives_gate(self):
        """极小容量（capacity=3）下 gate 后堆完整，可继续 add。"""
        from memory.working import WMItem, WorkingMemory
        wm = WorkingMemory(capacity=3)
        wm.add(_wm_item("a", "条目A高优", priority=0.9))
        wm.add(_wm_item("b", "条目B低优xyz", priority=0.1))
        wm.add(_wm_item("c", "条目C中优", priority=0.5))

        wm.salience_gate("今天天气", preserve_kinds=set(), priority_floor=0.7)

        # gate 后 add 不崩溃
        wm.add(WMItem(kind="new", content="新条目", priority=0.85))
        assert len(wm) <= 3

    def test_gate_with_all_preserve_kinds_drops_nothing(self):
        """所有条目均在 preserve_kinds 中→一条都不丢。"""
        from memory.working import TASK_SWITCH_PRESERVE_KINDS

        wm = self._wm()
        wm.add(_wm_item("bootstrap_identity", "身份", priority=0.1))
        wm.add(_wm_item("task_anchor", "任务", priority=0.1))

        dropped = wm.salience_gate(
            "随便什么消息",
            preserve_kinds=set(TASK_SWITCH_PRESERVE_KINDS),
            priority_floor=0.9,
        )
        assert dropped == 0
        assert len(wm) == 2

    def test_gate_keyword_boost_capped_at_1_for_already_high(self):
        """已接近 1.0 的条目 boost 后不超过 1.0。"""
        wm = self._wm()
        wm.add(_wm_item("task", "README 内容", priority=0.95))
        wm.salience_gate("README", preserve_kinds=set(), priority_floor=0.7, keyword_boost=0.3)
        for item in wm.get_top():
            assert item["priority"] <= 1.0

    def test_gate_ascii_keyword_case_insensitive(self):
        """ASCII 关键词大小写不敏感（_wm_keywords 会 lower）。"""
        from memory.working import _has_wm_overlap, _wm_keywords
        kws = _wm_keywords("check README Status")
        assert _has_wm_overlap("readme 进度已更新 status ok", kws)

    def test_gate_mixed_cjk_ascii_message(self):
        """中英混合消息：CJK ngram + ASCII 词同时起效。"""
        from memory.working import _has_wm_overlap, _wm_keywords
        kws = _wm_keywords("更新 README.md 进度")
        # CJK "进度" 命中
        assert _has_wm_overlap("任务进度 80%", kws)
        # ASCII "readme" 命中
        assert _has_wm_overlap("README file updated", kws)

    def test_gate_long_message_extracts_keywords(self):
        """长消息（> 50 字）应照常提取关键词。"""
        from memory.working import _wm_keywords
        msg = "请帮我把 lingzhou 项目的英文版 README.md 文件完成更新，重点写清楚安装步骤和配置说明"
        kws = _wm_keywords(msg)
        assert len(kws) > 5
        assert "readme" in kws or "README" in kws or any(k.lower() == "readme" for k in kws)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6：decide_continue 参数穿透验证
# ─────────────────────────────────────────────────────────────────────────────

class TestDecideContinuePassthrough:
    """decide_continue 应将 action_result / emotion_state 完整传给 assembler。"""

    @pytest.mark.asyncio
    async def test_passthrough_action_result_and_emotion(self):
        """验证 decide_continue 把 action_result/emotion_state 传入 assembler。"""
        from core.judgment.runtime import JudgmentLayer

        # 记录 _build_continue_context 被调用时的参数
        captured: dict = {}

        def _fake_build_ctx(self_inner, tool_history, *, user_message, reply_only,
                            wm_delta, speech_intent="", action_result=None, emotion_state=None):
            captured["action_result"] = action_result
            captured["emotion_state"] = emotion_state
            return "fake_ctx"

        ar = _make_ar(action_succeeded=True, tool_name="file.edit")
        emo = _make_emotion(dominant="focused", valence=0.6)

        # 构造最小 JudgmentLayer（绕过 __init__）
        jl = object.__new__(JudgmentLayer)

        # mock assembler
        asm = MagicMock()
        asm._last_context_text = "PREV"
        asm._build_continue_context = lambda *a, **kw: _fake_build_ctx(asm, *a, **kw)
        asm._build_messages = lambda ctx: []

        # mock executor
        async def _chat_with_retry(**kw):
            return None, MagicMock(tier="reasoner"), None

        executor = MagicMock()
        executor._select_provider.return_value = (MagicMock(), MagicMock(tier="reasoner"))
        executor._chat_with_retry = AsyncMock(return_value=(None, MagicMock(tier="reasoner"), None))
        executor._last_call_meta = {}

        jl._assembler = asm
        jl._executor = executor
        jl._cfg = MagicMock(thinking="low")

        # no-op _normalize_output
        async def _normalize(output, **kw):
            return output
        jl._normalize_output = _normalize

        result = await jl.decide_continue(
            [],
            user_message="hello",
            reply_only=True,
            action_result=ar,
            emotion_state=emo,
        )

        # executor 返回 None → JudgmentOutput.wait
        assert result is not None
        assert captured.get("action_result") is ar
        assert captured.get("emotion_state") is emo
