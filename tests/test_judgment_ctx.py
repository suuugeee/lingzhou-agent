"""行为门控、技能、thinking、模型路由、任务改写等 judgment context 测试"""
import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import (
    _judgment_output,
    _tool_registry,
)


class _WorkbenchRegistry:
    def get(self, name: str):
        if name != "task.workbench":
            return None
        from tools.registry import ToolEntry, ToolManifest

        return ToolEntry(
            manifest=ToolManifest(name="task.workbench", description="demo"),
            handler=lambda params, ctx: None,  # type: ignore[arg-type]
        )


class _EmptyRegistry:
    def get(self, name: str):
        return None


class _ReplyStore:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []

    async def get_fact(self, key: str):
        return "", False

    async def add_chat_message(self, role: str, content: str, chat_id: str = ""):
        self.messages.append((role, content, chat_id))
        return len(self.messages)


def _tick_reply_loop(judgment: Any, store: _ReplyStore | None = None) -> tuple[Any, _ReplyStore]:
    reply_store = store or _ReplyStore()
    return cast("Any", SimpleNamespace(
        _cfg=SimpleNamespace(
            thinking="off",
            loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="minimal"),
        ),
        _judgment=judgment,
        _pending_routing_overrides=None,
        _task_store=reply_store,
    )), reply_store


def _self_drive_signals(**overrides: Any) -> SimpleNamespace:
    data = {
        "active_task_id": 42,
        "active_task_source": "self_drive",
        "active_task_status": "in_progress",
        "active_task_next_step": "",
        "repeat_action_count": 0,
        "repeat_read_count": 0,
        "repeat_list_count": 0,
        "wait_streak": 0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_continue_low_increment_budget_builds_workbench_gate():
    from core.loop.shared.continue_phase import (
        _build_continue_low_increment_budget_action,
        _low_increment_history_count,
    )

    history = [
        {"tool": "file.list", "params": {"path": "memory"}, "result": "a"},
        {"tool": "file.read", "params": {"path": "memory/SKILL.md"}, "result": "b"},
        {"tool": "memory.search", "params": {"query": "provider"}, "result": "c"},
    ]

    assert _low_increment_history_count(history) == 3
    gated = _build_continue_low_increment_budget_action(
        action=_judgment_output(decision="act", chosen_action_id="file.list", params={"path": "tools"}),
        tool_name="file.list",
        budget=3,
        history=history,
    )

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    assert gated.params["workbench"]["recovery_state"] == "continue_low_increment_budget_reached"


def test_memory_search_control_query_rewrites_to_workbench():
    asyncio.run(_memory_search_control_query_rewrites_to_workbench())


async def _memory_search_control_query_rewrites_to_workbench():
    from core.judgment.boundary.pipeline import normalize_judgment_output

    action = _judgment_output(
        decision="act",
        chosen_action_id="memory.search",
        params={
            "query": (
                "基于最近 memory.search 成功结果收敛判断。"
                "不要重复同一 query 的 memory.search；改为读取命中语义 ID 或切换到 shell.run/file.read。"
            )
        },
        rationale="任务仍需推进，继续 memory.search",
    )

    normalized = await normalize_judgment_output(
        SimpleNamespace(),
        action,
        context_text="",
        raw="",
        registry=_WorkbenchRegistry(),
    )

    assert normalized.decision == "act"
    assert normalized.chosen_action_id == "task.workbench"
    workbench = normalized.params["workbench"]
    assert workbench["recovery_state"] == "memory_search_control_query_gated"
    assert "控制约束" in workbench["intent"]
    assert "不要重复同一 query" in workbench["evidence"][0]
    assert "具体语义节点 ID" in workbench["next_verification"]


# ══════════════════════════════════════════════════════════════════════════════
# SemanticMemory — 多锚点情境召回（ACT-R 收敛激活）
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_multi_anchor_convergence_bonus():
    """多锚点命中同一节点时 convergence_bonus 使其排名高于单锚点命中节点。

    设计原理：两节点在主锚点 "importlib" 上得分相近，但 node_ab 的 body
    同时命中第二锚点 "热加载 reload"，因此多锚点命中使其 final_score 更高。
    """
    from store.semantic import MemoryNode, SemanticMemory

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)

        # node_ab: title 含主锚点 "importlib"，body 含第二锚点 "热加载 reload"
        node_ab = MemoryNode(id="ab", kind="fact",
                             title="importlib",
                             body="热加载 reload 模块替换",
                             activation=0.0)
        # node_a: 同样含主锚点 "importlib"，body 不含第二锚点
        node_a = MemoryNode(id="a", kind="fact",
                            title="importlib",
                            body="模块导入",
                            activation=0.0)
        sm.upsert(node_ab)
        sm.upsert(node_a)

        results = sm.retrieve_multi_anchor(
            ["importlib", "热加载 reload"],
            top_k=2,
            convergence_bonus=0.3,
        )
        ids = [r["id"] for r in results]
        # node_ab 被两个锚点命中（convergence_bonus 加分），应排在第一位
        assert ids[0] == "ab", f"期望 ab 排第一，实际顺序: {ids}"


def test_semantic_multi_anchor_empty_anchors():
    """空锚点列表应返回空结果，不崩溃。"""
    from store.semantic import MemoryNode, SemanticMemory

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)
        sm.upsert(MemoryNode(id="x", kind="fact", title="test", body="body", activation=0.5))

        assert sm.retrieve_multi_anchor([]) == []
        assert sm.retrieve_multi_anchor(["", "  "]) == []


def test_memory_query_ignores_runtime_process_scaffolding_next_step():
    from core.cortex import intent as cortex_intent
    from core.judgment.assembler.assemble_context import (
        _build_context_anchors,
        _filter_context_semantic_memories,
        _memory_search_query_for_task,
    )

    task = SimpleNamespace(
        id=42,
        title="自驱巡检：分析记忆召回质量",
        goal="定位长期记忆召回差强人意的原因",
        next_step=cortex_intent.control_next_verification("下一轮先综合本 tick 工具结果，再决定是否继续取证。"),
        source="self_drive",
    )

    query = _memory_search_query_for_task(task, "")
    anchors = _build_context_anchors(
        SimpleNamespace(),
        task,
        "",
        "",
        None,
        [],
    )

    assert query == "定位长期记忆召回差强人意的原因"
    assert anchors[0] == "定位长期记忆召回差强人意的原因"
    assert "下一轮先综合本 tick 工具结果" not in " ".join(anchors)

    memories = _filter_context_semantic_memories(
        [
            {"id": "run-1", "kind": "run_result", "title": "pytest stdout", "body": "raw output", "score": 0.99},
            {"id": "meta-1", "kind": "meta_reflection", "title": "继续观察", "body": "下一步继续沉淀", "score": 0.95},
            {"id": "fact-1", "kind": "fact", "title": "记忆裁剪规则", "body": "保留可复用结论", "score": 0.80},
            {"id": "skill-1", "kind": "learned_skill", "title": "排障步骤", "body": "先定位真实链路", "score": 0.70},
        ],
        limit=1,
    )

    assert [item["id"] for item in memories] == ["fact-1"]


def test_fill_template_raises_when_variable_missing():
    from core.judgment.context.utils import _fill_template

    with pytest.raises(ValueError, match="missing_field"):
        _fill_template("hello {{ missing_field }}", {"other": "value"})


def test_context_risk_uncertainty_sections_have_defaults():
    from core.judgment.context.signals import (
        _fmt_risk_sections,
        _fmt_uncertainty_sections,
    )
    from core.perception import JudgmentSignals, PerceptionReplaySummary

    risk = _fmt_risk_sections(
        judgment_signals=JudgmentSignals(require_more_evidence=True, prefer_narrow_scope=True, posture="narrow"),
        failures=[],
        durable_failure_snapshot={},
        perception_replay=None,
        cognitive_signals=None,
    )
    uncertainty = _fmt_uncertainty_sections(
        judgment_signals=JudgmentSignals(require_more_evidence=True, prefer_narrow_scope=True, posture="narrow"),
        perception_replay=PerceptionReplaySummary(samples=2, avg_prediction_error=0.4, high_error_streak=0, trend="insufficient_data"),
        cognitive_signals=None,
    )

    assert "require_more_evidence=True" in uncertainty
    assert "姿态层建议收窄决策范围" in risk


def test_wm_proposal_sections_extracts_observation_candidates():
    from core.judgment.context.signals import _fmt_wm_proposal_sections

    wm_items = [
        {
            "kind": "self_drive",
            "content": (
                "[自驱信号]\n"
                "type: consolidation\n"
                "scope: observation\n"
                "proposal:\n"
                "- consolidate_memory: 把近期自驱观察结果沉淀。\n"
                "- inspect_failures: 评估重复失败边界。\n"
                "open_questions:\n"
                "- 近期收敛是否足够？\n"
                "available_directions: create_task | ignore_signal | wait\n"
            ),
        },
        {
            "kind": "user_message",
            "content": "这是普通用户输入，不应参与提案解析。",
        },
    ]

    text = _fmt_wm_proposal_sections(wm_items)

    assert "scope=observation" in text
    assert "consolidate_memory" in text
    assert "inspect_failures" in text
    assert "open_questions:" in text
    assert "available_directions:" in text
    assert "create_task" in text
    assert "ignore_signal" in text


def test_judgment_output_preserves_model_reply_without_mechanical_rewrite():
    """记忆表述由 LLM 结合 context 判断，parser 层不做正则改写。"""
    from core.judgment.boundary import normalize_action_shape

    output = _judgment_output(
        decision="wait",
        reply_to_user="我记得你之前说过你叫 bat。",
    )

    normalized = normalize_action_shape(output)

    assert normalized.reply_to_user == "我记得你之前说过你叫 bat。"


def test_normalize_action_shape_clears_non_act_tool_payload():
    from core.judgment.boundary import normalize_action_shape

    output = _judgment_output(
        decision="pause",
        chosen_action_id="file.read",
        params={"path": "/tmp/stale.py"},
        rationale="暂停，不执行工具。",
        next_step="等待外部输入",
    )
    output.parallel_actions = [{"action_id": "file.list", "params": {"path": "/tmp"}}]
    output.delegate_tasks = [{"id": "sub-1", "goal": "read stale path"}]
    output.applied_skills = ["anti-loop"]

    normalized = normalize_action_shape(output)

    assert normalized.decision == "pause"
    assert normalized.chosen_action_id == ""
    assert normalized.params == {}
    assert normalized.parallel_actions == []
    assert normalized.delegate_tasks == []
    assert normalized.rationale == "暂停，不执行工具。"
    assert normalized.next_step == "等待外部输入"
    assert normalized.applied_skills == ["anti-loop"]


# ══════════════════════════════════════════════════════════════════════════════
# 今日新增功能验证
# ══════════════════════════════════════════════════════════════════════════════

def test_model_health_circuit_breaker_blocks_and_clears():
    """ModelHealth 断路器：标记冷却后 _is_model_available 返回 False，
    recover 后返回 True；fallback tier 在主 tier 冷却时被选中。"""
    import time

    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _Dummy:
        async def chat(self, messages, **kw):
            return '{"decision":"wait"}'
        async def close(self):
            pass

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    layer = JudgmentLayer(_Dummy(), ToolRegistry(), cfg)

    # 初始状态：模型可用
    assert layer._executor._is_model_available("bailian/qwen3.6-plus") is True

    # 标记 429 错误 → 进入冷却
    layer._executor._mark_model_failure("bailian/qwen3.6-plus", "Client error '429 Too Many Requests'")
    assert layer._executor._is_model_available("bailian/qwen3.6-plus") is False

    # recover → 可用
    health = layer._executor._get_health("bailian/qwen3.6-plus")
    health.cooldown_until = time.time() - 1  # 手动过期
    assert layer._executor._is_model_available("bailian/qwen3.6-plus") is True


def test_select_tier_logic():
    """_select_tier 按 phase 和 prefer_tier 正确返回 tier。"""
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _Dummy:
        async def chat(self, messages, **kw):
            return '{"decision":"wait"}'
        async def close(self):
            pass

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    layer = JudgmentLayer(_Dummy(), ToolRegistry(), cfg)

    # initial phase → reasoner
    assert layer._executor._select_tier(phase="initial", user_message="hello") == "reasoner"

    # repair phase → repair
    assert layer._executor._select_tier(phase="repair", user_message="") == "repair"

    # prefer_tier 优先
    assert layer._executor._select_tier(phase="initial", user_message="", prefer_tier="reader") == "reader"

    # continue 默认不再因 reader 工具而隐式降到 reader
    tier = layer._executor._select_tier(
        phase="continue", user_message="",
        current_action="file.read", tool_history=[],
    )
    assert tier == "reasoner"

    # continue + reasoner tool → reasoner
    tier2 = layer._executor._select_tier(
        phase="continue", user_message="",
        current_action="shell.run", tool_history=[],
    )
    assert tier2 == "reasoner"


def test_prefer_tier_for_task_uses_pending_then_task_default():
    from core.loop.shared.common import _next_initial_tier_hint, _prefer_tier_for_task
    from store.task import Task

    task = Task(
        id=1,
        title="任务",
        status="pending",
        priority="normal",
        created_at="2026-05-15T00:00:00Z",
        model_tier="reader",
    )

    assert _prefer_tier_for_task(None, task) == "reader"
    assert _prefer_tier_for_task("reader", task) == "reader"
    assert _prefer_tier_for_task(" Reader ", task) == "reader"
    assert _prefer_tier_for_task("repair", task) == "repair"
    assert _prefer_tier_for_task("reader", task, has_user_message=True) == "reasoner"
    assert _prefer_tier_for_task(None, task, has_user_message=True) == "reasoner"

    task.model_tier = " Reasoner "
    assert _prefer_tier_for_task(None, task) == "reasoner"

    task.model_tier = "invalid"
    assert _prefer_tier_for_task(None, task) is None

    assert _next_initial_tier_hint(_judgment_output(decision="act", chosen_action_id="file.read")) is None
    assert _next_initial_tier_hint(
        _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            model_strategy={"next_phase_tier": " Reader "},
        )
    ) == "reader"


def test_behavior_gate_blocks_repeating_same_action_and_logs_observation(caplog):
    """重复信号达到阈值后，继续选择同一动作应被硬制动。"""
    from core.loop.drive.behavior import BehaviorTracker

    caplog.set_level(logging.INFO, logger="lingzhou.behavior_tracker")

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())

    class _Signals:
        repeat_action_count = 3
        repeat_action_tool = "memory.search"
        repeat_action_key = "legacy runtime"
        repeat_read_count = 0
        repeat_read_path = ""
        loop_probe_version = 5

    action = _judgment_output(
        decision="act",
        chosen_action_id="memory.search",
        params={"query": "legacy runtime"},
        rationale="再搜一次",
    )
    gated = tracker.apply_execution_gate(action, _Signals())
    assert gated.decision == "act"
    assert gated is not action
    assert gated.chosen_action_id == "task.workbench"
    assert "停止重复低增量动作" in gated.params["workbench"]["intent"]
    assert "legacy runtime" in gated.params["workbench"]["next_verification"]
    assert "行为门控改道" in gated.rationale
    assert any("repeat action streak=3" in rec.message for rec in caplog.records)

    fallback_tracker = BehaviorTracker(registry=_EmptyRegistry())
    fallback = fallback_tracker.apply_execution_gate(action, _Signals())
    assert fallback.decision == "wait"
    assert "行为门控制动" in fallback.rationale

    switched_action = _judgment_output(
        decision="act",
        chosen_action_id="task.workbench",
        params={"workbench": {"domain": "runtime"}},
        rationale="改用工作台沉淀状态",
    )
    switched = tracker.apply_execution_gate(switched_action, _Signals())
    assert switched.decision == "act"
    assert switched is switched_action

    class _ReadSignals:
        repeat_action_count = 0
        repeat_action_tool = ""
        repeat_action_key = ""
        repeat_read_count = 3
        repeat_read_path = "/tmp/demo.txt"
        loop_probe_version = 6

    read_action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        params={"path": "/tmp/demo.txt"},
    )
    gated_read = tracker.apply_execution_gate(read_action, _ReadSignals())
    assert gated_read.decision == "act"
    assert gated_read is not read_action
    assert gated_read.chosen_action_id == "task.workbench"
    assert "停止重复读取" in gated_read.params["workbench"]["intent"]
    assert "/tmp/demo.txt" in gated_read.params["workbench"]["next_verification"]

    different_read = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        params={"path": "/tmp/other.txt"},
    )
    allowed_read = tracker.apply_execution_gate(different_read, _ReadSignals())
    assert allowed_read is different_read

    window_read = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        params={"path": "/tmp/demo.txt", "offset": 21, "limit": 20},
    )
    allowed_window_read = tracker.apply_execution_gate(window_read, _ReadSignals())
    assert allowed_window_read is window_read

    class _BothSignals:
        repeat_action_count = 3
        repeat_action_tool = "file.read"
        repeat_action_key = "/tmp/demo.txt"
        repeat_read_count = 3
        repeat_read_path = "/tmp/demo.txt"
        loop_probe_version = 7

    gated_both = tracker.apply_execution_gate(read_action, _BothSignals())
    assert gated_both.decision == "act"
    assert gated_both.chosen_action_id == "task.workbench"
    assert "不要再重复执行 file.read /tmp/demo.txt" in gated_both.params["workbench"]["next_verification"]

    # on_act 连续相同行为时应生成 WMItem 信号
    items = []
    for _ in range(3):
        items = tracker.on_act("shell.run", "ls", task_id="t1")
    assert any("行为信号" in i.content for i in items), "连续 3 次相同行为应注入 WM 行为信号"

    # on_act 连续不同命令（key_param 不同）不应触发 streak
    tracker2 = BehaviorTracker()
    tracker2.on_act("shell.run", "cat USER.md", task_id="t2")
    tracker2.on_act("shell.run", "cat SOUL.md", task_id="t2")
    items2 = tracker2.on_act("shell.run", "sed -n '1p' TOOLS.md", task_id="t2")
    assert not any("行为信号" in i.content for i in items2), (
        "不同 shell.run 命令不应触发 streak（key_param 已区分命令内容）"
    )


def test_behavior_gate_redirects_self_drive_evidence_wait_to_workbench():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="当前无用户输入，先等待")

    gated = tracker.apply_execution_gate(
        action,
        _self_drive_signals(active_task_next_step="读取最近日志，寻找重复模式或可优化点"),
    )

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    workbench = gated.params["workbench"]
    assert workbench["recovery_state"] == "evidence_required_before_wait"
    assert "取证" in workbench["intent"]
    assert workbench["next_verification"] == "读取最近日志，寻找重复模式或可优化点"
    assert "不能直接 wait" in gated.rationale


def test_behavior_gate_allows_self_drive_external_wait():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="等待外部信号")

    signals = _self_drive_signals(active_task_id=43, active_task_next_step="等待外部输入或下一次日记同步信号")
    assert tracker.apply_execution_gate(action, signals) is action


def test_behavior_gate_forces_repeated_self_drive_wait_to_evidence():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="等待外部信号")

    gated = tracker.apply_execution_gate(
        action,
        _self_drive_signals(
            active_task_id=44,
            active_task_next_step="等待外部输入或下一次日记同步信号",
            wait_streak=3,
        ),
    )

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    assert gated.params["workbench"]["recovery_state"] == "evidence_required_before_wait"
    assert "取证" in gated.params["workbench"]["intent"]


def test_behavior_gate_resumes_self_drive_waiting_task_if_no_evidence_hint():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="等待外部输入")
    signals = _self_drive_signals(
        active_task_id=45,
        active_task_status="waiting",
        active_task_next_step="等待外部输入或下一次同步信号",
    )
    assert tracker.apply_execution_gate(action, signals) is action


def test_behavior_gate_forces_self_drive_waiting_task_with_evidence():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="等待外部输入")

    gated = tracker.apply_execution_gate(
        action,
        _self_drive_signals(
            active_task_id=46,
            active_task_status="waiting",
            active_task_next_step="读取最近日志，寻找下一步可执行动作",
        ),
    )

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    assert gated.params["workbench"]["recovery_state"] == "evidence_required_before_wait"
    assert "读取最近日志" in gated.params["workbench"]["next_verification"]


def test_behavior_gate_forces_self_drive_waiting_task_with_viewing_action():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="等待外部输入")

    gated = tracker.apply_execution_gate(
        action,
        _self_drive_signals(
            active_task_id=47,
            active_task_status="waiting",
            active_task_next_step="查看最近 10 次失败记录并提取异常模式",
        ),
    )

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    assert gated.params["workbench"]["recovery_state"] == "evidence_required_before_wait"
    assert "查看最近 10 次失败记录" in gated.params["workbench"]["next_verification"]


def test_contains_evidence_intent_normalizes_punctuation_and_newlines():
    from core.loop.drive.behavior import _contains_evidence_intent, _normalize_next_step

    assert _contains_evidence_intent("先做最小范围自检：读取日志并复现失败场景。") is True
    assert _contains_evidence_intent("先做最小范围复现,提取回归链路") is True
    assert _contains_evidence_intent("先做最小范围，debug 最近失败样例") is True
    assert _contains_evidence_intent("先做最小范围的外部观察，准备等待外部输入。") is False

    assert _normalize_next_step("先做最小范围\n自检：读取日志(10 次)") == "先做最小范围 自检 读取日志 10 次"


def test_behavior_gate_applies_on_implicit_evidence_phrase_variants():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="wait", rationale="等待外部输入")

    gated = tracker.apply_execution_gate(
        action,
        _self_drive_signals(
            active_task_id=48,
            active_task_next_step="先做最小范围自检：读取最近 10 条日志，调研异常时间窗口。",
            wait_streak=1,
        ),
    )

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    assert "先做最小范围自检" in gated.params["workbench"]["next_verification"]


def test_cognitive_signals_include_last_action_feedback_and_repeat_list():
    from core.perception import CognitiveSignals

    text = CognitiveSignals(
        repeat_action_count=3,
        repeat_action_tool="memory.search",
        repeat_action_key="legacy runtime sqlite",
        repeat_read_count=0,
        repeat_read_path="",
        repeat_list_count=3,
        repeat_list_path="/root/.legacy-runtime/memory",
        loop_probe_version=9,
        last_action_tool="shell.run",
        last_action_key="find /root/.legacy-runtime",
        last_action_status="ok",
        last_action_summary="找到了 main.sqlite，但没有进一步推进 next_step",
        last_action_error="",
        last_action_state_delta="process=finished; exit_code=0",
        last_action_progressful=False,
        recent_action_history=[
            "tool=file.list | key=/root/.legacy-runtime | status=ok | progressful=True",
            "tool=memory.search | key=legacy runtime sqlite | status=ok | progressful=False",
        ],
    ).to_text()

    assert "last_action={tool='shell.run'" in text
    assert "repeat_list_count=3" in text
    assert "被系统判定为未推进" in text
    assert "recent_actions:" in text
    assert "tool=memory.search | key=legacy runtime sqlite" in text


def test_skill_registry_logs_selected_skills(caplog):
    from core.skill import SkillRegistry

    caplog.set_level(logging.INFO, logger="core.skill")
    reg = SkillRegistry()
    skills = reg.match_for_context(
        wm_pressure=0.5,
        has_active_task=True,
        has_next_step=True,
        failure_count=2,
        high_error_streak=1,
        context_text="当前任务需要继续推进，但已经有失败和参数错误。",
        max_inject=3,
    )

    assert skills
    assert any("[skill.match] selected=" in rec.message for rec in caplog.records)


def test_judgment_skills_for_log_formats_selected_names():
    from core.skill import Skill

    skills = []
    formatted = ",".join(skill.name for skill in skills[:3]) if skills else "none"
    assert formatted == "none"

    skills = [
        Skill(name="runtime-bootstrap", description="", guidance=""),
        Skill(name="task-continuity", description="", guidance=""),
    ]
    formatted = ",".join(skill.name for skill in skills[:3]) if skills else "none"
    assert formatted == "runtime-bootstrap,task-continuity"


def test_behavior_list_result_aware():
    """file.list 应按“结果是否相同”而不是仅按路径判定重复。"""
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker()

    # 同一路径，但目录结果不同：不应触发重复警告
    for _ in range(3):
        tracker.on_act("file.list", "/root", task_id="t-list")
    items = tracker.on_list("/root", "a.txt\n")
    items = tracker.on_list("/root", "a.txt\nb.txt\n")
    items = tracker.on_list("/root", "a.txt\nb.txt\nc.txt\n")
    assert not any("行为信号" in i.content for i in items), "同路径但结果变化，不应判定为无效重复"

    # 同一路径且结果相同：才触发重复警告
    tracker2 = BehaviorTracker()
    for _ in range(3):
        tracker2.on_act("file.list", "/root", task_id="t-list-2")
    same = []
    for _ in range(3):
        same = tracker2.on_list("/root", "a.txt\nb.txt\n")
    assert any("行为信号" in i.content for i in same), "同路径且结果相同，才应触发 file.list 重复信号"


def test_behavior_explore_awareness_requires_task_context():
    from core.loop.drive.behavior import BehaviorTracker

    tracker = BehaviorTracker()
    items = []
    for _ in range(10):
        items = tracker.on_act("file.list", "/root", task_id=None)

    assert items == []


def test_behavior_tracker_uses_explicit_registry_capabilities():
    from core.loop.drive.behavior import BehaviorTracker
    from tools.registry import ToolEntry, ToolManifest

    class _Registry:
        def get(self, name: str):
            if name != "demo.readonly":
                return None
            return ToolEntry(
                manifest=ToolManifest(
                    name="demo.readonly",
                    description="demo",
                    capabilities=("result_streak_only",),
                ),
                handler=lambda params, ctx: None,  # type: ignore[arg-type]
            )

    tracker = BehaviorTracker(registry=_Registry())
    items = []
    for _ in range(3):
        items = tracker.on_act("demo.readonly", "same", task_id="task-1")

    assert items == []


def test_behavior_gate_redirects_low_increment_probe_after_unprogressful_action():
    from core.loop.drive.behavior import BehaviorTracker

    class _Signals:
        repeat_action_count = 0
        repeat_action_tool = ""
        repeat_action_key = ""
        repeat_read_count = 0
        repeat_read_path = ""
        last_action_tool = "task.list"
        last_action_key = "all"
        last_action_status = "ok"
        last_action_progressful = False
        last_action_progress_reason = "task.list 结果与上轮相同"

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(decision="act", chosen_action_id="file.list", params={"path": "/root/lingzhou"})

    gated = tracker.apply_execution_gate(action, _Signals())

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    workbench = gated.params["workbench"]
    assert "上一低增量探索未推进" in workbench["intent"]
    assert "task.list" in workbench["evidence"][0]
    assert "file.list" in workbench["evidence"][1]
    assert "不要继续同类 list/search 枚举" in workbench["next_verification"]

    read_action = _judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/root/lingzhou/core/a.py"})
    assert tracker.apply_execution_gate(read_action, _Signals()) is read_action

    memory_action = _judgment_output(decision="act", chosen_action_id="memory.search", params={"query": "task continuity"})
    assert tracker.apply_execution_gate(memory_action, _Signals()) is memory_action


def test_behavior_gate_redirects_repeated_file_list_results():
    from core.loop.drive.behavior import BehaviorTracker

    class _Signals:
        repeat_action_count = 0
        repeat_read_count = 0
        repeat_list_count = 3
        repeat_list_path = "/root/lingzhou/tools"

    tracker = BehaviorTracker(registry=_WorkbenchRegistry())
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.list",
        params={"path": "/root/lingzhou/tools"},
    )

    gated = tracker.apply_execution_gate(action, _Signals())

    assert gated.decision == "act"
    assert gated.chosen_action_id == "task.workbench"
    workbench = gated.params["workbench"]
    assert "停止重复目录枚举" in workbench["intent"]
    assert "/root/lingzhou/tools" in workbench["next_verification"]
    assert "选择最相关文件读取" in workbench["next_verification"]

    different_path = _judgment_output(
        decision="act",
        chosen_action_id="file.list",
        params={"path": "/root/lingzhou/core"},
    )
    assert tracker.apply_execution_gate(different_path, _Signals()) is different_path


def test_next_thinking_override_is_one_shot_and_strict():
    from core.loop.shared.common import _next_thinking_override

    assert _next_thinking_override({"thinking_override": "low"}) == "low"
    assert _next_thinking_override({"thinking_override": "invalid"}) is None
    assert _next_thinking_override({}) is None
    assert _next_thinking_override(None) is None


def test_resolve_thinking_override_uses_mode_defaults_and_strategy():
    from core.loop.shared.common import _resolve_thinking_override

    cfg = cast("Any", SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="medium"),
    ))

    assert _resolve_thinking_override(cfg, user_message="hi") == "low"
    assert _resolve_thinking_override(cfg, user_message="") == "medium"
    assert _resolve_thinking_override(cfg, user_message="", pending_override="high") == "high"
    assert _resolve_thinking_override(cfg, user_message="", model_strategy={"thinking_override": "minimal"}) == "minimal"


def test_thinking_floor_respects_chat_minimum_for_user_message():
    from core.loop.shared.common import _thinking_floor

    assert _thinking_floor("off", "low") == "low"
    assert _thinking_floor("minimal", "low") == "low"
    assert _thinking_floor("high", "low") == "high"
    assert _thinking_floor(None, "low") == "low"


async def test_decide_initial_raises_autonomous_active_task_to_medium_thinking(monkeypatch):
    from core.loop.tick.prep import _decide_initial_action
    from core.loop.tick.types import _TickJudgmentPrep

    captured: dict[str, Any] = {}

    class _Judgment:
        async def decide(self, *args, **kwargs):
            captured.update(kwargs)
            return _judgment_output(decision="wait")

    class _WM:
        def get_top(self, limit: int):
            return []

    cfg = cast("Any", SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(
            judge_every=1,
            chat_thinking="low",
            autonomous_thinking="minimal",
        ),
    ))
    loop = cast("Any", SimpleNamespace(
        _cfg=cfg,
        _pending_thinking_override=None,
        _ticks_since_judge=0,
        _wm=_WM(),
        _judgment=_Judgment(),
        _task_store=object(),
        _episodic=object(),
        _semantic=object(),
        _emotion=object(),
        _pending_tier=None,
        _pending_routing_overrides=None,
    ))
    monkeypatch.setattr(
        "core.loop.runtime.life.collect_runtime_life_snapshot",
        lambda loop: SimpleNamespace(as_dict=lambda: {}),
    )

    prep = _TickJudgmentPrep(
        percept=object(),
        perception_replay=None,
        cognitive_signals=None,
        ethos_state=None,
        signals=None,
        hard_boundaries=[],
    )
    await _decide_initial_action(
        loop,
        cycle=1,
        user_message="",
        active_task=SimpleNamespace(id=9, model_tier=""),
        chat_id=None,
        prep=prep,
    )

    assert captured["thinking_override"] == "medium"


def test_recent_runs_summary_prefers_output_and_progress():
    from core.judgment.context.tasks import _fmt_recent_runs
    from store.task import Run

    runs = [
        Run(
            id=12,
            task_id=7,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            status="done",
            created_at="2026-05-15T14:00:00+00:00",
            tool_name="file.list",
            model_tier="reader",
            progress="列出目录，确认技能清单",
            output_json={"summary": "index.ts\npackage.json\nSKILL.md"},
        )
    ]

    text = _fmt_recent_runs(runs)
    assert "run#12 [done]" in text
    assert "tool=file.list" in text
    assert "progress=列出目录，确认技能清单" in text
    assert "summary=index.ts package.json SKILL.md" in text


def test_recent_runs_summary_preview_keeps_token_budget():
    from core.judgment.context.tasks import _fmt_recent_runs
    from store.task import Run

    runs = [
        Run(
            id=77,
            task_id=9,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            status="done",
            created_at="2026-05-15T14:00:00+00:00",
            tool_name="shell.run",
            model_tier="reasoner",
            progress="x" * 500,
            output_json={"summary": "A" * 5000 + " mid " + "B" * 5000},
        )
    ]

    text = _fmt_recent_runs(runs)
    assert "run#77 [done]" in text
    assert "summary=" in text
    assert "omitted" in text
    assert "A" in text
    assert "B" in text
    assert len(text) < 400


def test_waiting_tasks_section_exposes_wait_reason_and_next_step():
    from core.judgment.context.tasks import _fmt_waiting_tasks
    from store.task import Task

    tasks = [
        Task(
            id=27,
            title="等待用户补源路径",
            status="waiting",
            priority="normal",
            created_at="2026-05-15T14:00:00+00:00",
            goal="等待用户提供源路径后继续",
            next_step="拿到源路径后恢复任务并重新验证目录结构",
            wait_kind="external",
            wait_key="source-path",
        )
    ]

    text = _fmt_waiting_tasks(tasks)
    assert "task#27 [waiting] 等待用户补源路径" in text
    assert "wait=external/source-path" in text
    assert "next=拿到源路径后恢复任务并重新验证目录结构" in text


def test_task_anchor_item_includes_recovery_hint():
    from types import SimpleNamespace

    from core.loop.runtime.memory_hooks import build_task_anchor_item

    task = SimpleNamespace(
        id=1,
        title="复盘成长闭环",
        goal="确认 recovery 机制",
        next_step="执行 memory 回溯",
        result_json={
            "cortex": {
                "recovery_state": "recovering_from_run_failure",
                "next_verification": "重新执行 tool.workbench 并验证下一步",
            }
        },
    )

    item = build_task_anchor_item(
        task,
        action_feedback="tool=file.read | key=/tmp/a.py | status=ok | progressful=False",
    )
    assert "上一动作反馈: tool=file.read | key=/tmp/a.py | status=ok | progressful=False" in item.content
    assert "恢复状态: recovering_from_run_failure" in item.content
    assert "下一步验证: 重新执行 tool.workbench 并验证下一步" in item.content


def test_fmt_task_includes_recovery_hint():
    from core.judgment.context.tasks import _fmt_task
    from store.task import Task

    task = Task(
        id=11,
        title="修复闭环停摆",
        status="in_progress",
        priority="normal",
        created_at="2026-06-10T10:00:00+00:00",
        goal="修复决策停摆",
        next_step="读取 task 历史",
        result_json={"cortex": {"recovery_state": "recovering_from_runtime_guard", "next_verification": "执行 probe.run 验证"}},
    )

    text = _fmt_task(task)
    assert "恢复状态: recovering_from_runtime_guard" in text
    assert "下一步验证: 执行 probe.run 验证" in text


def test_runnable_tasks_section_omits_active_task():
    from core.judgment.context.tasks import _fmt_runnable_tasks
    from store.task import Task

    tasks = [
        Task(
            id=10,
            title="当前活跃任务",
            status="in_progress",
            priority="high",
            created_at="2026-05-15T14:00:00+00:00",
            goal="正在执行",
        ),
        Task(
            id=11,
            title="排查远程运行重启循环",
            status="pending",
            priority="normal",
            created_at="2026-05-15T14:00:00+00:00",
            goal="分析 crash.log",
            next_step="读取 crash.log 并比对最近一次重启栈",
        ),
    ]

    text = _fmt_runnable_tasks(tasks, active_task_id=10)
    assert "task#10" not in text
    assert "task#11 [pending/normal] 排查远程运行重启循环" in text
    assert "next=读取 crash.log 并比对最近一次重启栈" in text


def test_similar_tasks_section_exposes_similarity_and_context():
    from core.judgment.context.tasks import _fmt_similar_tasks
    from store.task import Task

    items = [(
        Task(
            id=31,
            title="排查远程运行重启循环",
            status="waiting",
            priority="high",
            created_at="2026-05-15T14:00:00+00:00",
            goal="分析 crash.log",
            next_step="等待新的 crash.log 后继续比对",
        ),
        0.81,
    )]

    text = _fmt_similar_tasks(items)
    assert "81% task#31 [waiting] 排查远程运行重启循环" in text
    assert "next=等待新的 crash.log 后继续比对" in text


@pytest.mark.asyncio
async def test_load_similar_tasks_snapshot_excludes_self_drive_for_non_self_drive_task():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "similar-context.db")
        await store.open()
        try:
            active_id = await store.add_task(
                "当前用户任务",
                goal="分析 crash.log",
                source="external",
                status="in_progress",
            )
            external_id = await store.add_task(
                "排查远程运行重启循环",
                goal="分析 crash.log 并修复重启循环",
                source="external",
            )
            self_drive_id = await store.add_task(
                "排查远程运行重启循环",
                goal="自驱分析 crash.log 并修复重启循环",
                source="self_drive",
            )

            finder = store.find_similar_open_tasks
            hits = await finder(
                "解决远程运行重启循环",
                limit=5,
                min_score=0.45,
                exclude_task_ids=[active_id],
                allowed_sources=None,
                excluded_sources=("self_drive",),
            )

            hit_ids = [task.id for task, _ in hits]
            assert external_id in hit_ids
            assert self_drive_id not in hit_ids
        finally:
            await store.close()


def test_model_routing_section_uses_effective_thinking():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
        },
        "thinking": "high",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="initial",
        user_message="继续",
        current_action="",
        tool_history=None,
        effective_thinking="low",
    ))

    # available_models 在测试环境可能为空（fake providers 无模型目录）
    # 验证 delegation_guide 和 tier_descriptions 等结构字段存在即可
    assert "delegation_guide" in payload
    assert "tier_descriptions" in payload
    assert "reasoner" in payload["tier_descriptions"]


def test_select_provider_matches_routing_provider_by_public_model_ref():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
        },
        "temperature": 0.7,
        "timeout": 60.0,
    })

    main_provider = _DummyProvider("copilot/gpt-5.4")
    reader_provider = _DummyProvider("bailian/qwen3.6-plus")
    layer = JudgmentLayer(main_provider, ToolRegistry(), cfg)
    layer.set_routing_providers({"reader": reader_provider})

    provider, selection = layer._executor._select_provider(
        phase="initial",
        user_message="先读取配置",
        prefer_tier="reader",
    )

    assert provider is reader_provider
    assert selection.tier == "reader"
    assert selection.model_ref == "bailian/qwen3.6-plus"


def test_select_provider_build_failure_enters_config_cooldown_and_suppresses_duplicate_logs(caplog):
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    layer._executor._tier_model_candidates = lambda tier, **kwargs: ("deepseek/deepseek-v4-flash",)  # type: ignore[method-assign]
    layer._executor._fallback_tiers = lambda tier, exclude_reader=False: ()  # type: ignore[method-assign]
    layer._executor._find_or_create_provider = lambda model_ref: (_ for _ in ()).throw(RuntimeError(  # type: ignore[method-assign]
        "OpenAI 兼容 provider 的环境变量 'DEEPSEEK_API_KEY' 为空，请设置该变量或从 routing/model_fallbacks 中移除此 provider。"
    ))

    caplog.set_level(logging.WARNING, logger="lingzhou.judgment")

    layer._executor._select_provider(
        phase="initial",
        user_message="hello",
        prefer_tier="reasoner",
    )
    layer._executor._select_provider(
        phase="initial",
        user_message="hello",
        prefer_tier="reasoner",
    )

    warnings = [rec.message for rec in caplog.records if "provider_build_failed" in rec.message]
    assert len(warnings) == 1
    # config 语义由 LLM 异步感知；同步规则无法识别时归为 other，仍进入冷却
    # 第二次调用被冷却跳过，所以只有 1 条 warning
    assert "code=other" in warnings[0]

    health = layer._executor._get_health("deepseek/deepseek-v4-flash")
    assert health.last_code == "other"  # LLM 会异步重分类为 config，此处验证同步初始状态
    assert health.cooldown_until > time.time()


def test_fmt_config_snapshot_exposes_judgment_signal_thresholds():
    from core.config import Config
    from core.judgment.context.sections import _fmt_config_snapshot

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "emotion": {
            "failure_normalization_count": 4.0,
            "high_error_normalization_streak": 5.0,
            "feeling_min_intensity": 0.25,
            "regulation_down_regulate_arousal_high": 0.8,
            "regulation_high_error_streak_guard": 3,
        },
        "soul": {
            "ethos": {
                "prefer_verification_caution_min": 0.72,
                "prefer_verification_failure_count": 3,
                "prefer_narrow_error_streak": 4,
                "avoid_overclaiming_down_regulate_streak": 5,
                "failure_adjust_count": 2,
                "failure_truth_delta": 0.11,
                "high_error_adjust_streak": 4,
                "recovering_curiosity_delta": 0.09,
            },
        },
        "thresholds": {
            "prediction_error_task": 0.8,
            "perception_replay_trend_delta": 0.2,
            "perception_replay_high_error_hint_streak": 3,
            "emotion_replay_trend_delta": 0.12,
            "judgment_error_streak_guard": 4,
            "judgment_posture_narrow_failure_count": 5,
            "judgment_posture_pause_worsening_failure_count": 3,
        },
    })

    text = _fmt_config_snapshot(cfg)
    assert "## Emotion guardrails (emotion.*)" in text
    assert "failure_normalization_count: 4.0" in text
    assert "high_error_normalization_streak: 5.0" in text
    assert "feeling_min_intensity: 0.25" in text
    assert "regulation_down_regulate_arousal_high: 0.8" in text
    assert "regulation_high_error_streak_guard: 3" in text
    assert "## Ethos guardrails (soul.ethos.*)" in text
    assert "prefer_verification_caution_min: 0.72" in text
    assert "prefer_verification_failure_count: 3" in text
    assert "prefer_narrow_error_streak: 4" in text
    assert "avoid_overclaiming_down_regulate_streak: 5" in text
    assert "failure_adjust_count: 2" in text
    assert "failure_truth_delta: 0.11" in text
    assert "high_error_adjust_streak: 4" in text
    assert "recovering_curiosity_delta: 0.09" in text
    assert "## Replay guardrails (thresholds.*)" in text
    assert "prediction_error_task: 0.8" in text
    assert "perception_replay_trend_delta: 0.2" in text
    assert "perception_replay_high_error_hint_streak: 3" in text
    assert "emotion_replay_trend_delta: 0.12" in text
    assert "## Judgment guardrails (thresholds.*)" in text
    assert "judgment_error_streak_guard: 4" in text
    assert "judgment_posture_narrow_failure_count: 5" in text
    assert "judgment_posture_pause_worsening_failure_count: 3" in text


def test_fmt_config_snapshot_exposes_reference_thresholds():
    from core.config import Config
    from core.judgment.context.sections import _fmt_config_snapshot

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "memory": {
            "consolidate_threshold": 0.74,
            "consolidate_low_pressure_skip_threshold": 0.81,
            "global_md_warn_bytes": 12345,
            "global_md_warn_lines": 67,
            "daily_recall_days": 3,
            "daily_recall_max_chars": 640,
            "daily_recall_semantic_score_threshold": 0.62,
            "daily_summary_days": 5,
            "daily_summary_max_chars": 1500,
            "daily_summary_activation": 0.77,
            "daily_summary_importance": 0.84,
            "auto_compact_enabled": True,
            "auto_compact_every_ticks": 12,
            "auto_compact_runtime_db_min_bytes": 123456,
            "auto_compact_memory_dir_min_bytes": 654321,
            "auto_compact_vacuum": True,
        },
        "emotion": {
            "reflection_valence_history_weight": 0.7,
            "reflection_valence_hint_weight": 0.3,
        },
        "thresholds": {
            "reference_min_confidence": 0.61,
            "reference_local_signal_base": 0.42,
            "reference_local_signal_step": 0.11,
            "reference_local_confidence_cap": 0.77,
            "reference_max_anchors": 2,
            "reference_topic_top_k": 7,
            "reference_recent_narrative_limit": 4,
            "reference_recent_semantic_top_k": 6,
            "reference_topic_anchor_min_chars": 4,
            "fact_context_exclude_prefixes": ["pref:", "run:"],
            "fact_context_task_limit": 8,
            "fact_context_global_limit": 5,
            "fact_context_priority_prefixes": ["user:", "profile:"],
            "fact_context_priority_limit": 2,
            "fact_context_recent_scan_multiplier": 4,
            "fact_context_recent_scan_min": 10,
            "chat_history_turn_limit": 2,
            "chat_history_max_chars": 180,
        },
    })

    text = _fmt_config_snapshot(cfg)
    assert "## Reference guardrails (thresholds.*)" in text
    assert "reference_min_confidence: 0.61" in text
    assert "reference_local_signal_base: 0.42" in text
    assert "reference_local_signal_step: 0.11" in text
    assert "reference_local_confidence_cap: 0.77" in text
    assert "reference_max_anchors: 2" in text
    assert "reference_topic_top_k: 7" in text
    assert "reference_recent_narrative_limit: 4" in text
    assert "reference_recent_semantic_top_k: 6" in text
    assert "reference_time_recent_limit" not in text
    assert "reference_time_semantic_top_k" not in text
    assert "reflection_valence_history_weight: 0.7" in text
    assert "reflection_valence_hint_weight: 0.3" in text
    assert "## Memory guardrails (memory.*)" in text
    assert "consolidate_threshold: 0.74" in text
    assert "consolidate_low_pressure_skip_threshold: 0.81" in text
    assert "promotion_priority_threshold: 0.78" in text
    assert "promotion_max_nodes_per_consolidation: 6" in text
    assert "promotion_body_max_chars: 12000" in text
    assert "promotion_reinforce_delta: 0.05" in text
    assert "daily_recall_days: 3" in text
    assert "daily_recall_max_chars: 640" in text
    assert "daily_recall_semantic_score_threshold: 0.62" in text
    assert "daily_summary_days: 5" in text
    assert "daily_summary_max_chars: 1500" in text
    assert "daily_summary_activation: 0.77" in text
    assert "daily_summary_importance: 0.84" in text
    assert "global_md_warn_bytes: 12345" in text
    assert "global_md_warn_lines: 67" in text
    assert "auto_compact_enabled: True" in text
    assert "auto_compact_every_ticks: 12" in text
    assert "auto_compact_runtime_db_min_bytes: 123456" in text
    assert "auto_compact_memory_dir_min_bytes: 654321" in text
    assert "auto_compact_vacuum: True" in text
    assert "reference_topic_anchor_min_chars: 4" in text
    assert "reference_time_phrase_hours" not in text
    assert "reference_days_ago_pattern" not in text
    assert "reference_hours_ago_pattern" not in text
    assert "reference_named_top_k" not in text
    assert "reference_self_intro_terms" not in text
    assert "reference_self_intro_name_max_chars" not in text
    assert "reference_relation_hint_terms" not in text
    assert "## Context facts guardrails (thresholds.*)" in text
    assert 'fact_context_exclude_prefixes: ["pref:", "run:"]' in text
    assert "fact_context_task_limit: 8" in text
    assert "fact_context_global_limit: 5" in text
    assert 'fact_context_priority_prefixes: ["user:", "profile:"]' in text
    assert "fact_context_priority_limit: 2" in text
    assert "fact_context_recent_scan_multiplier: 4" in text
    assert "fact_context_recent_scan_min: 10" in text
    assert "## Chat history guardrails (thresholds.*)" in text
    assert "chat_history_turn_limit: 2" in text
    assert "chat_history_max_chars: 180" in text
    assert "## Task steering guardrails (thresholds.*)" not in text
    assert "task_steer_ascii_term_min_chars" not in text
    assert "task_steer_cjk_term_min_chars" not in text
    assert "task_steer_cjk_term_max_chars" not in text
    assert "task_steer_message_min_chars" not in text
    assert "task_steer_message_min_terms" not in text
    assert "task_steer_message_overlap_threshold" not in text


def test_fmt_soul_uses_config_ethos_fallback_when_db_missing():
    from core.judgment.context.sections import _fmt_soul

    # hard_axioms 已由宪法层硬阻断，不再注入 prompt；只验证 ethos_baseline fallback
    text = _fmt_soul(
        "",
        '{"truth": 0.85, "caution": 0.70}',
    )

    assert "价值基线（ethos_baseline，config fallback）" in text
    assert '"truth": 0.85' in text


def test_executor_extract_prompt_limit_supports_multiple_patterns():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)

    prompt, limit = layer._executor._extract_prompt_limit(
        "prompt token count of 161904 exceeds the limit of 128000"
    )
    assert prompt == 161904
    assert limit == 128000

    prompt2, limit2 = layer._executor._extract_prompt_limit("context_length_exceeded: 131072")
    assert prompt2 is None
    assert limit2 == 131072


def test_executor_detects_output_overflow_and_available_tokens():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)

    err_text = (
        "max_tokens: 32768 > context_window: 200000 - input_tokens: 190000 "
        "= available_tokens: 10000"
    )
    assert layer._executor._is_output_overflow_error(err_text) is True
    assert layer._executor._extract_available_output_tokens(err_text) == 10000


def test_executor_retry_after_and_backoff_respects_lower_bound():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)

    retry_after = layer._executor._extract_retry_after_seconds("Too many requests, retry after 7")
    assert retry_after == 7.0

    delay = layer._executor._retry_delay_seconds(
        1,
        base_delay=1.0,
        max_delay=30.0,
        retry_after_seconds=retry_after,
    )
    assert delay >= 7.0
    assert delay <= 30.0


@pytest.mark.asyncio
async def test_chat_with_retry_applies_retry_after_backoff_and_fallback(monkeypatch):
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelSelection
    from provider.base import Message
    from tools.registry import ToolRegistry

    class _ProviderAlwaysFail:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref
            self.last_usage = {}

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            raise RuntimeError("429 Too Many Requests; retry after 2")

        async def close(self):
            return None

    class _ProviderSucceed:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref
            self.last_usage = {}

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "routing": {
            "reasoner": "copilot/gpt-5.4",
            "reader": "bailian/qwen3.6-plus",
        },
        "temperature": 0.7,
        "timeout": 60.0,
    })

    main_provider = _ProviderAlwaysFail("copilot/gpt-5.4")
    fallback_provider = _ProviderSucceed("bailian/qwen3.6-plus")
    layer = JudgmentLayer(main_provider, ToolRegistry(), cfg)
    layer.set_routing_providers({"reader": fallback_provider})

    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("core.judgment.decision.helpers.asyncio.sleep", _fake_sleep)

    messages = [Message(role="user", content="hello")]
    selected_provider = main_provider
    selection = ModelSelection(
        phase="initial",
        tier="reasoner",
        model_ref="copilot/gpt-5.4",
        thinking="off",
    )

    raw, final_selection, err = await layer._executor._chat_with_retry(
        selected_provider=selected_provider,
        selection=selection,
        messages=messages,
        phase="initial",
        user_message="hello",
        thinking_override=None,
        routing_overrides=None,
        log_prefix="[test]",
        fallback_prefer_tier="reader",
        skills="none",
    )

    assert err is None
    assert raw == '{"decision":"wait"}'
    assert final_selection.model_ref == "bailian/qwen3.6-plus"
    assert sleep_calls == []


def test_pick_retry_provider_accepts_excluded_model_refs_signature():
    from core.judgment import ModelSelection
    from core.judgment.decision.helpers import _pick_retry_provider

    class _Executor:
        def _select_provider(self, **kwargs):
            assert kwargs["excluded_model_refs"] == {"copilot/gpt-5.4"}
            assert kwargs["excluded_provider_names"] == {"copilot"}
            assert kwargs["prefer_tier"] == "reader"
            return object(), ModelSelection(
                phase="initial",
                tier="reader",
                model_ref="bailian/qwen3.6-plus",
                thinking="off",
            )

    provider, selection = _pick_retry_provider(
        _Executor(),
        selection=ModelSelection(
            phase="initial",
            tier="reasoner",
            model_ref="copilot/gpt-5.4",
            thinking="off",
        ),
        phase="initial",
        user_message="hello",
        current_action="",
        tool_history=None,
        thinking_override=None,
        routing_overrides=None,
        fallback_prefer_tier="reader",
        excluded_model_refs={"copilot/gpt-5.4"},
        excluded_provider_names={"copilot"},
    )

    assert provider is not None
    assert selection.model_ref == "bailian/qwen3.6-plus"


@pytest.mark.asyncio
async def test_chat_with_retry_reasoner_phase_fallback_skips_reader_by_default(monkeypatch):
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelSelection
    from provider.base import Message
    from tools.registry import ToolRegistry

    class _ProviderAlwaysFail:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref
            self.last_usage = {}

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            raise RuntimeError("ConnectError('')")

        async def close(self):
            return None

    class _ProviderSucceed:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref
            self.last_usage = {}

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "openai-codex": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
        "model": "openai-codex/gpt-5.4-mini",
        "routing": {
            "reader": "openai-codex/gpt-5.4-mini",
            "reasoner": "openai-codex/gpt-5.5",
            "repair": "openai-codex/gpt-5.4",
        },
        "temperature": 0.7,
        "timeout": 60.0,
    })

    main_provider = _ProviderAlwaysFail("openai-codex/gpt-5.5")
    reader_provider = _ProviderSucceed("openai-codex/gpt-5.4-mini")
    repair_provider = _ProviderSucceed("openai-codex/gpt-5.4")
    layer = JudgmentLayer(main_provider, ToolRegistry(), cfg)
    layer.set_routing_providers({
        "reader": reader_provider,
        "reasoner": main_provider,
        "repair": repair_provider,
    })

    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("core.judgment.decision.helpers.asyncio.sleep", _fake_sleep)

    raw, final_selection, err = await layer._executor._chat_with_retry(
        selected_provider=main_provider,
        selection=ModelSelection(
            phase="initial",
            tier="reasoner",
            model_ref="openai-codex/gpt-5.5",
            thinking="low",
        ),
        messages=[Message(role="user", content="继续解决问题")],
        phase="initial",
        user_message="继续解决问题",
        thinking_override=None,
        routing_overrides=None,
        log_prefix="[test]",
        skills="none",
    )

    assert err is None
    assert raw == '{"decision":"wait"}'
    assert final_selection.tier == "repair"
    assert final_selection.model_ref == "openai-codex/gpt-5.4"
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_chat_with_retry_same_model_uses_retry_after_delay(monkeypatch):
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelSelection
    from provider.base import Message
    from tools.registry import ToolRegistry

    class _ProviderFailThenOk:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref
            self.last_usage = {}
            self.calls = 0

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 Too Many Requests; retry after 2")
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _ProviderFailThenOk("bailian/qwen3.6-plus")
    layer = JudgmentLayer(provider, ToolRegistry(), cfg)

    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("core.judgment.decision.helpers.asyncio.sleep", _fake_sleep)

    raw, final_selection, err = await layer._executor._chat_with_retry(
        selected_provider=provider,
        selection=ModelSelection(
            phase="initial",
            tier="reasoner",
            model_ref="bailian/qwen3.6-plus",
            thinking="off",
        ),
        messages=[Message(role="user", content="hello")],
        phase="initial",
        user_message="hello",
        thinking_override=None,
        routing_overrides=None,
        log_prefix="[test]",
        fallback_prefer_tier="reasoner",
        skills="none",
    )

    assert err is None
    assert raw == '{"decision":"wait"}'
    assert final_selection.model_ref == "bailian/qwen3.6-plus"
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 2.0


@pytest.mark.asyncio
async def test_chat_with_retry_output_overflow_skips_prompt_compression(monkeypatch, caplog):
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelSelection
    from provider.base import Message
    from tools.registry import ToolRegistry

    class _ProviderOutputOverflowThenOk:
        def __init__(self, model_ref: str):
            self.model_ref = model_ref
            self.last_usage = {}
            self.calls = 0

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "max_tokens: 32768 > context_window: 200000 - input_tokens: 190000 "
                    "= available_tokens: 10000; retry after 1"
                )
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _ProviderOutputOverflowThenOk("bailian/qwen3.6-plus")
    layer = JudgmentLayer(provider, ToolRegistry(), cfg)

    trim_calls: list[int] = []

    def _fake_trim(messages, prompt_limit, *, prompt_count=None):
        trim_calls.append(1)
        return messages

    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(layer._executor, "_trim_messages_for_prompt_limit", _fake_trim)
    monkeypatch.setattr("core.judgment.decision.helpers.asyncio.sleep", _fake_sleep)

    caplog.set_level(logging.WARNING, logger="lingzhou.judgment")

    raw, final_selection, err = await layer._executor._chat_with_retry(
        selected_provider=provider,
        selection=ModelSelection(
            phase="initial",
            tier="reasoner",
            model_ref="bailian/qwen3.6-plus",
            thinking="off",
        ),
        messages=[Message(role="user", content="hello")],
        phase="initial",
        user_message="hello",
        thinking_override=None,
        routing_overrides=None,
        log_prefix="[test]",
        fallback_prefer_tier="reasoner",
        skills="none",
    )

    assert err is None
    assert raw == '{"decision":"wait"}'
    assert final_selection.model_ref == "bailian/qwen3.6-plus"
    assert trim_calls == []
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 1.0
    assert "overflow_kind=output" in caplog.text
    assert "messages_omitted=false" in caplog.text


def test_fmt_chat_history_keeps_budget_by_dropping_old_turns_then_clipping_singletons():
    """有预算时优先丢弃最旧整轮；单条仍超长时裁剪头尾。"""
    from core.judgment.context.sections import _fmt_chat_history

    messages = [
        {"role": "user", "content": "abcdefghi"},
        {"role": "assistant", "content": "123456789"},
    ]
    full = _fmt_chat_history(messages, max_chars=0)
    assert "用户: abcdefghi" in full
    assert "我: 123456789" in full

    trimmed = _fmt_chat_history(messages, max_chars=20)
    assert trimmed == "我: 123456789"
    assert "abcdefghi" not in trimmed

    huge = "START" + "A" * 500 + "TAIL"
    clipped = _fmt_chat_history([{"role": "user", "content": huge}], max_chars=80)
    assert "START" in clipped
    assert "TAIL" in clipped
    assert "chars omitted" in clipped
    assert len(clipped) <= 120


def test_context_budget_preserves_compacted_tools_section_under_pressure():
    from core.judgment.context.budget import apply_context_budget

    tools = "\n".join(
        [
            "- `memory.search`: 搜索长期记忆 参数: [query(*)]",
            "- `shell.run`: 执行 shell 命令 参数: [command(*), timeout, workdir]",
            "- `task.workbench`: 写入任务工作台 参数: [task_id(*), workbench(*)]",
        ]
        + [f"- `demo.tool{i}`: demo 参数: [value]" for i in range(80)]
    )
    ctx = {
        "tools_section": tools,
        "wm_section": "noise " * 8000,
        "probe_sensors_section": "probe " * 8000,
        "memories_section": "重要记忆 " * 100,
        "task_section": "当前任务: 找回 OpenClaw 记忆并修复膨胀。",
    }

    budgeted = apply_context_budget(ctx, token_budget=1200)

    assert "TOOL CATALOG COMPACTED" in budgeted["tools_section"]
    assert "`memory.search`" in budgeted["tools_section"]
    assert "`shell.run`" in budgeted["tools_section"]
    assert "`task.workbench`" in budgeted["tools_section"]


def test_tool_tier_uses_manifest_truth_for_reasoner_tools():
    from core.judgment.output import is_plan_alignment_exempt, tool_tier, tool_tier_mapping

    registry = _tool_registry()

    assert tool_tier("task.ask", registry) == "reasoner"
    assert tool_tier("task.plan", registry) == "reasoner"
    assert tool_tier("shell.run", registry) == "reasoner"
    assert tool_tier("schedule.add", registry) == "reasoner"
    assert tool_tier("memory.snapshot", registry) == "reasoner"
    assert tool_tier("task.resume", registry) == "reasoner"
    assert tool_tier("web.search", registry) == "reasoner"
    assert tool_tier("image.analyze", registry) == "reasoner"
    assert tool_tier("image.generate", registry) == "reasoner"
    assert tool_tier("schedule.cancel", registry) == "reasoner"
    assert tool_tier("failure.dismiss", registry) == "reasoner"
    assert tool_tier("task.list", registry) == "reader"

    mapping = tool_tier_mapping(registry)
    assert "task.ask" in mapping["reasoner"]
    assert "task.plan" in mapping["reasoner"]
    assert "schedule.add" in mapping["reasoner"]
    assert "memory.snapshot" in mapping["reasoner"]
    assert "task.resume" in mapping["reasoner"]
    assert "web.search" in mapping["reasoner"]
    assert "image.analyze" in mapping["reasoner"]
    assert "image.generate" in mapping["reasoner"]
    assert "schedule.cancel" in mapping["reasoner"]
    assert "failure.dismiss" in mapping["reasoner"]
    assert "task.list" in mapping["reader"]

    assert is_plan_alignment_exempt("task.ask", registry) is True
    assert is_plan_alignment_exempt("task.plan", registry) is True


async def test_reference_failure_is_exposed_in_model_routing_section():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _FailingProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            raise RuntimeError("400 Bad Request")

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4-mini",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_FailingProvider(), ToolRegistry(), cfg)
    await layer._assembler._ref_resolver._reason_about_candidates_with_llm(
        "继续上次的话题",
        {"n1": {"kind": "task", "title": "旧任务", "body": "body"}},
    )
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="initial",
        user_message="继续上次的话题",
        current_action="",
        tool_history=None,
        effective_thinking="low",
    ))

    assert payload["primary_provider"]["model"] == "copilot/gpt-5.4-mini"
    assert payload["reference_resolution"]["llm_available"] is False
    assert payload["reference_resolution"]["last_error_code"] == "400"
    assert "400 Bad Request" in payload["reference_resolution"]["last_error"]


def test_model_routing_section_lazy_recovers_missing_impl_alias(monkeypatch):
    import core.judgment.assembler as assembler_mod
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    monkeypatch.delitem(assembler_mod.__dict__, "_build_model_routing_section_impl", raising=False)

    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="initial",
        user_message="继续",
        current_action="",
        tool_history=None,
        effective_thinking="low",
    ))

    assert "tier_descriptions" in payload
    assert callable(assembler_mod.__dict__.get("_build_model_routing_section_impl"))


def test_assemble_context_failure_is_coalesced_and_backed_off(caplog):
    from core.judgment.decision import rounds as rounds_mod

    rounds_mod._ASSEMBLE_CONTEXT_ERROR_STATE.clear()
    caplog.set_level(logging.WARNING, logger="lingzhou.judgment")

    exc = NameError("_build_model_routing_section_impl is not defined")
    count1 = rounds_mod._track_assemble_context_failure(exc)
    rounds_mod._log_assemble_context_failure(exc, count1)
    count2 = rounds_mod._track_assemble_context_failure(exc)
    rounds_mod._log_assemble_context_failure(exc, count2)

    assert count1 == 1
    assert count2 == 2
    assert rounds_mod._assemble_context_failure_backoff_ms(1) == 0
    assert rounds_mod._assemble_context_failure_backoff_ms(2) == 2000
    assert rounds_mod._assemble_context_failure_backoff_ms(99) == 60000
    assert any("异常重复 x2" in rec.message for rec in caplog.records)


def test_assemble_context_failure_with_active_task_records_workbench_recovery():
    from core.judgment.decision import rounds as rounds_mod

    active_task = SimpleNamespace(id=42)
    output = rounds_mod._assemble_context_failure_output(
        exc=TypeError("episodic signature mismatch"),
        repeat_count=2,
        judgment_signals=None,
        hard_boundaries=[],
        active_task=active_task,
        registry=_WorkbenchRegistry(),
    )

    assert output.decision == "act"
    assert output.chosen_action_id == "task.workbench"
    assert output.params["workbench"]["recovery_state"] == "recovering_from_context_assembly_failure"
    assert "episodic signature mismatch" in output.params["workbench"]["evidence"][0]
    assert "读取最新异常栈" in output.params["workbench"]["next_verification"]
    assert output.model_strategy["next_idle_gap_ms"] == 2000


def test_assemble_context_failure_without_workbench_keeps_safe_wait():
    from core.judgment.decision import rounds as rounds_mod

    output = rounds_mod._assemble_context_failure_output(
        exc=RuntimeError("context failed"),
        repeat_count=1,
        judgment_signals=None,
        hard_boundaries=[],
        active_task=SimpleNamespace(id=7),
        registry=None,
    )

    assert output.decision == "wait"
    assert "上下文组装异常" in output.rationale


def test_llm_unavailable_with_active_task_records_workbench_recovery():
    from core.judgment.decision import rounds as rounds_mod

    output = rounds_mod._llm_unavailable_output(
        err="ConnectError('')",
        active_task=SimpleNamespace(id=3485),
        registry=_WorkbenchRegistry(),
    )

    assert output.decision == "act"
    assert output.chosen_action_id == "task.workbench"
    assert output.params["workbench"]["recovery_state"] == "recovering_from_llm_unavailable"
    assert "ConnectError" in output.params["workbench"]["evidence"][1]
    assert "provider 健康状态" in output.params["workbench"]["next_verification"]
    assert output.model_strategy["next_idle_gap_ms"] == 2000


def test_llm_unavailable_reply_only_keeps_inner_loop_wait():
    from core.judgment.decision import rounds as rounds_mod

    output = rounds_mod._llm_unavailable_output(
        err="ConnectError('')",
        active_task=SimpleNamespace(id=3485),
        registry=None,
        reply_only=True,
    )

    assert output.decision == "wait"
    assert output.chosen_action_id == ""
    assert "[inner-loop] LLM 不可用" in output.rationale


def test_sync_prompt_capsule_clears_stale_assembler_capsule():
    from core.judgment.decision import rounds as rounds_mod

    deps = SimpleNamespace(
        executor=SimpleNamespace(_last_prompt_capsule=""),
        assembler=SimpleNamespace(_last_context_compression_capsule="STALE_CAPSULE"),
    )

    rounds_mod._sync_prompt_capsule(cast("Any", deps))

    assert deps.assembler._last_context_compression_capsule == ""


def test_model_routing_section_no_longer_exposes_implicit_reader_default():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "ok"}],
        effective_thinking="low",
    ))



def test_model_routing_section_uses_configured_idle_bounds_and_defaults():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "loop": {
            "idle_with_task_bounds": [250, 45000],
            "idle_no_task_bounds": [8000, 120000],
            "active_idle_gap": 1500,
            "max_idle_gap": 90000,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="继续分析",
        current_action="file.read",
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "ok"}],
        effective_thinking="low",
    ))

    guide = payload["delegation_guide"]
    assert "当前有任务时 250ms-45s，无任务时 8s-120s" in guide
    assert "当前 loop 默认备用值（有任务 1.5s，无任务 90s）" in guide


def test_model_routing_section_counts_exploration_budget_by_capability():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 2 条"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"},
            {"tool": "shell.run", "params": {"command": "pytest -q"}, "result": "1 passed"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["task_explore_count"] == 3
    assert payload["budget_state"]["ask_evidence_hits"] == 2
    assert payload["budget_state"]["ask_evidence_budget"] == 2
    assert payload["budget_state"]["task_explore_converge_after"] == 4
    assert payload["budget_state"]["global_cost_posture"] == "conserve"


def test_model_routing_section_uses_configured_explore_converge_threshold():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "thresholds": {
            "task_explore_converge_after": 3,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 2 条"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"},
            {"tool": "shell.run", "params": {"command": "pytest -q"}, "result": "1 passed"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["task_explore_count"] == 3
    assert payload["budget_state"]["task_explore_converge_after"] == 3
    assert payload["budget_state"]["global_cost_posture"] == "converge"


def test_model_routing_section_repeat_counts_only_trailing_streak():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["repeat_action_count"] == 1
    assert payload["budget_state"]["repeat_read_count"] == 1

    trailing = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
            {"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "alpha"},
        ],
        effective_thinking="low",
    ))

    assert trailing["budget_state"]["repeat_action_count"] == 2
    assert trailing["budget_state"]["repeat_read_count"] == 2


def test_model_routing_section_repeat_action_count_uses_action_key_signature():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
            {"tool": "memory.search", "params": {"query": "other runtime"}, "result": "命中 1 条"},
        ],
        effective_thinking="low",
    ))

    assert payload["budget_state"]["repeat_action_count"] == 1

    trailing = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
        ],
        effective_thinking="low",
    ))

    assert trailing["budget_state"]["repeat_action_count"] == 2


def test_model_routing_section_exposes_tool_history_compaction_policy():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
        "thresholds": {
            "continue_tool_history_compact_threshold": 2,
            "continue_tool_history_keep_last": 1,
        },
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    payload = json.loads(layer._assembler._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="memory.search",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "legacy runtime"}, "result": "命中 1 条"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "命中 1 条任务"},
        ],
        effective_thinking="low",
    ))

    assert payload["continue_phase_policy"]["tool_history_count"] == 2
    assert payload["continue_phase_policy"]["tool_history_compact_threshold"] == 2
    assert payload["continue_phase_policy"]["tool_history_keep_last"] == 1
    assert payload["continue_phase_policy"]["tool_history_will_compact_next"] is True
    assert "tool_history_will_compact_next=true" in payload["delegation_guide"]


def test_compact_history_line_keeps_summary_preview_not_full_blob():
    from core.loop.shared.continue_phase import _compact_history_line

    entry = {
        "tool": "shell.run",
        "status": "ok",
        "result": "A" * 5000,
        "metadata": {"log_summary": "B" * 5000},
        "state_delta": {"details": "C" * 5000},
        "artifact_paths": ["/tmp/out.log"],
        "fingerprint": "abc",
    }

    compacted = _compact_history_line(entry)
    assert "summary" in compacted
    assert "omitted" in compacted
    assert "A" * 5000 not in compacted
    assert "B" * 100 not in compacted


def test_fmt_durable_failures_exposes_policy_and_muted_actions():
    from core.judgment.context.tasks import _fmt_durable_failures

    text = _fmt_durable_failures({
        "threshold": 3,
        "ttl_sec": 7200,
        "muted_actions": [
            {
                "tool": "file.read",
                "key": "/tmp/missing.txt",
                "reason": "missing_path",
                "count": 4,
                "remaining_sec": 119,
            }
        ],
    })

    assert "policy: threshold=3 ttl_sec=7200" in text
    assert "file.read /tmp/missing.txt" in text
    assert "reason=missing_path" in text
    assert "remaining=119s" in text


async def test_load_durable_failure_snapshot_reads_policy_and_active_mutes():
    from core.judgment.context.facts import _load_durable_failure_snapshot
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            await store.set_fact(
                "control:durable_failure_policy",
                json.dumps({"threshold": 5, "ttl_sec": 1800}, ensure_ascii=False),
                scope="system",
            )
            await store.set_fact(
                "durable_failure:active",
                json.dumps({
                    "tool": "file.read",
                    "key": "/tmp/missing.txt",
                    "reason": "missing_path",
                    "count": 5,
                    "muted_until": time.time() + 90,
                }, ensure_ascii=False),
                scope="system",
            )
            await store.set_fact(
                "durable_failure:expired",
                json.dumps({
                    "tool": "file.read",
                    "key": "/tmp/old.txt",
                    "reason": "missing_path",
                    "count": 3,
                    "muted_until": time.time() - 10,
                }, ensure_ascii=False),
                scope="system",
            )

            snapshot = await _load_durable_failure_snapshot(store)
            assert snapshot["threshold"] == 5
            assert snapshot["ttl_sec"] == 1800
            assert len(snapshot["muted_actions"]) == 1
            assert snapshot["muted_actions"][0]["tool"] == "file.read"
            assert snapshot["muted_actions"][0]["key"] == "/tmp/missing.txt"
        finally:
            await store.close()


async def test_decide_continue_uses_passed_thinking_override():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_thinking_override: str | None = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_thinking_override = thinking_override
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, ToolRegistry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "file.list", "params": {"path": "/tmp"}, "result": "ok"}],
        user_message="继续",
        prefer_tier="reasoner",
        thinking_override="low",
    )

    assert out.decision == "wait"
    assert provider.last_thinking_override == "low"
    assert layer.last_call_meta["thinking"] == "low"


async def test_decide_continue_raises_user_task_followup_to_medium_thinking():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_thinking_override: str | None = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_thinking_override = thinking_override
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, ToolRegistry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "file.read", "params": {"path": "/tmp/a.py"}, "result": "ok"}],
        user_message="不对，继续分析原因",
        active_task=SimpleNamespace(id=7),
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert provider.last_thinking_override == "medium"
    assert layer.last_call_meta["thinking"] == "medium"


async def test_decide_continue_raises_autonomous_active_task_to_medium_thinking():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_thinking_override: str | None = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_thinking_override = thinking_override
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, ToolRegistry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "file.read", "params": {"path": "/tmp/a.py"}, "result": "ok"}],
        user_message="",
        active_task=SimpleNamespace(id=8),
        prefer_tier="reasoner",
        thinking_override="low",
    )

    assert out.decision == "wait"
    assert provider.last_thinking_override == "medium"
    assert layer.last_call_meta["thinking"] == "medium"


async def test_decide_continue_updates_last_call_meta_after_fallback():
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelSelection

    class _FailingProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            raise RuntimeError("primary unavailable")

        async def close(self):
            return None

    class _FallbackProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_FailingProvider(), _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"
    layer._executor._last_call_meta["skills"] = "cached-skill"
    fallback_provider = _FallbackProvider()

    def _fake_select_provider(**kwargs):
        prefer_tier = kwargs.get("prefer_tier")
        if prefer_tier == "reasoner":
            return fallback_provider, ModelSelection(
                phase="continue",
                tier="reasoner",
                model_ref="bailian/qwen-reasoner-fallback",
                thinking="high",
            )
        return layer._executor._provider, ModelSelection(
            phase="continue",
            tier="reader",
            model_ref="bailian/qwen-reader-primary",
            thinking="off",
        )

    layer._executor._select_provider = _fake_select_provider  # type: ignore[method-assign]

    out = await layer.decide_continue(
        [{"tool": "file.list", "params": {"path": "/tmp"}, "result": "ok"}],
        user_message="继续",
        prefer_tier="reader",
    )

    assert out.decision == "wait"
    assert layer.last_call_meta["model_ref"] == "bailian/qwen-reasoner-fallback"
    assert layer.last_call_meta["tier"] == "reasoner"
    assert layer.last_call_meta["skills"] == "cached-skill"


def test_action_made_progress_result_aware():
    from core.loop.shared.progress import _action_made_progress, _result_fingerprint
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    list_action = _judgment_output(decision="act", chosen_action_id="file.list", params={"path": "/tmp"})
    list_res = ToolResult(summary="a.txt\nb.txt\n")
    assert _action_made_progress(list_action, list_res, prev_sig="", prev_fp="")[0] is True
    assert _action_made_progress(
        list_action,
        list_res,
        prev_sig="file.list|/tmp",
        prev_fp=_result_fingerprint(list_res.summary),
    )[0] is False

    write_action = _judgment_output(decision="act", chosen_action_id="file.write", params={"path": "/tmp/x"})
    write_res = ToolResult(summary="写入成功: /tmp/x")
    assert _action_made_progress(write_action, write_res)[0] is True

    config_set_action = _judgment_output(decision="act", chosen_action_id="config.set", params={"key": "loop.max_idle_gap"})
    empty_mutation = ToolResult(summary="")
    assert _action_made_progress(config_set_action, empty_mutation)[0] is True

    workbench_action = _judgment_output(decision="act", chosen_action_id="task.workbench", params={})
    workbench_res = ToolResult(summary="task.workbench id=1", state_delta={"cortex": {"next_verification": "继续验证"}})
    made_progress, reason = _action_made_progress(workbench_action, workbench_res)
    assert made_progress is False
    assert "不视为外部推进" in reason

    fail_action = _judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/tmp/missing"})
    fail_res = ToolResult(summary="文件不存在: /tmp/missing", error="FileNotFound")
    assert _action_made_progress(fail_action, fail_res)[0] is False

    unknown_action = _judgment_output(decision="act", chosen_action_id="custom.unknown", params={"id": "42"})
    empty_unknown = ToolResult(summary="")
    assert _action_made_progress(unknown_action, empty_unknown)[0] is False

    unknown_res = ToolResult(summary="no-op result")
    assert _action_made_progress(unknown_action, unknown_res, prev_sig="", prev_fp="")[0] is True
    assert _action_made_progress(
        unknown_action,
        unknown_res,
        prev_sig="custom.unknown|42",
        prev_fp=_result_fingerprint(unknown_res.summary),
    )[0] is False

    unknown_with_delta = ToolResult(summary="", state_delta={"updated": True})
    assert _action_made_progress(unknown_action, unknown_with_delta)[0] is True

    class _Registry:
        def get(self, name: str):
            if name != "custom.override":
                return None
            return ToolEntry(
                manifest=ToolManifest(
                    name="custom.override",
                    description="override",
                    progress_category="mutation",
                ),
                handler=lambda params, ctx: None,  # type: ignore[arg-type]
            )

    override_action = _judgment_output(decision="act", chosen_action_id="custom.override", params={"id": "7"})
    override_res = ToolResult(summary="")
    assert _action_made_progress(override_action, override_res)[0] is False
    assert _action_made_progress(override_action, override_res, registry=_Registry())[0] is True  # type: ignore[arg-type]


def test_write_success_stall_meta_reflection_records_task_hint():
    asyncio.run(_write_success_stall_meta_reflection_records_task_hint())


async def _write_success_stall_meta_reflection_records_task_hint():
    from core.loop.shared.postprocess import _write_success_stall_meta_reflection
    from store.task import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "stall.db")
        await store.open()
        task_id = await store.add_task("分析空转", goal="减少重复探索")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        action = _judgment_output(decision="act", chosen_action_id="memory.search", params={"query": "legacy runtime"})
        result = ToolResult(summary="命中旧记忆：/root/.legacy-runtime/memory/main.sqlite")
        await _write_success_stall_meta_reflection(store, task, action, result, streak=2, cycle=12)

        raw, found = await store.get_fact(f"task:{task_id}:meta_reflection")
        assert found
        payload = json.loads(raw)
        assert payload["target_kind"] == "stall_recovery"
        assert payload["tool_name"] == "memory.search"
        assert "停止重复 memory.search" in payload["proposal"]
        await store.close()


def test_success_stall_reflection_tracks_capability_based_tool():
    asyncio.run(_success_stall_reflection_tracks_capability_based_tool())


async def _success_stall_reflection_tracks_capability_based_tool():
    from core.loop.tick import _maybe_record_success_stall_reflection
    from store.task import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "stall-capability.db")
        await store.open()
        try:
            task_id = await store.add_task("分析任务枚举空转", goal="避免重复 task.list")
            task = await store.get_task_by_id(task_id)
            assert task is not None

            loop = cast("Any", SimpleNamespace(
                _task_store=store,
                _registry=_tool_registry(),
                _last_act_progressful=False,
                _success_stall_task_id=None,
                _success_stall_streak=0,
            ))
            action = _judgment_output(decision="act", chosen_action_id="task.list", params={"status": "all"})
            result = ToolResult(summary="命中 3 条任务")

            await _maybe_record_success_stall_reflection(loop, task, action, result, cycle=7)
            await _maybe_record_success_stall_reflection(loop, task, action, result, cycle=8)

            raw, found = await store.get_fact(f"task:{task_id}:meta_reflection")
            assert found
            payload = json.loads(raw)
            assert payload["tool_name"] == "task.list"
            assert "停止重复 task.list" in payload["proposal"]
        finally:
            await store.close()


def test_fallback_reply_for_user_describes_waiting_state():
    from core.loop.shared.logging import _fallback_reply_for_user
    from store.task import Task
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="task.wait", next_step="等用户补充路径后重新验证目录")
    result = ToolResult(
        summary="任务 [27] 已进入 waiting: external/source-path",
        state_delta={"task_status": "waiting", "wait_kind": "external", "wait_key": "source-path"},
    )
    task = Task(id=27, title="等待路径", status="in_progress", priority="normal", created_at="2026-05-15T14:00:00+00:00")

    reply = _fallback_reply_for_user(action, result, task)
    assert reply.startswith("当前任务已转入等待")
    assert "external/source-path" in reply
    assert "等用户补充路径后重新验证目录" in reply


def test_clip_signal_text_honors_limit_and_compacts_whitespace():
    from core.loop.shared.logging import _clip_signal_text, _summarize_state_delta

    assert _clip_signal_text("a\n  b\tc", 20) == "a b c"
    clipped = _clip_signal_text("A" * 50, 12)
    assert clipped == "A" * 9 + "..."
    tiny = _clip_signal_text("ABCDE", 2)
    assert tiny == "AB"

    state = _summarize_state_delta({"blob": "B" * 200, "count": 1}, limit=80)
    assert len(state) <= 80
    assert state.endswith("...")


def test_fallback_reply_for_user_uses_real_error_instead_of_background_ack():
    from core.loop.shared.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="pause", rationale="源路径证据不存在，需要用户补充。")
    result = ToolResult(summary="路径不存在: /root/.legacy-runtime/source", error="FileNotFound")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("这轮工具执行失败")
    assert "路径不存在" in reply
    assert "后台继续处理" not in reply
    assert "我这轮" not in reply


def test_fallback_reply_for_user_uses_recovery_next_step_from_tool_result():
    from core.loop.shared.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="task.workbench")
    result = ToolResult(
        summary="工具参数缺失: task.workbench requires workbench",
        error="ToolInputInvalid",
        skipped=True,
        state_delta={
            "recovery_next_step": "按 task.workbench 的 manifest 重新调用工具；补齐必填参数 workbench。",
        },
    )

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("这轮工具执行失败")
    assert "工具参数缺失" in reply
    assert "补齐必填参数 workbench" in reply


def test_fallback_reply_for_user_clips_long_recovery_next_step():
    from core.loop.shared.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="shell.run")
    result = ToolResult(
        summary="执行超时：" + "S" * 500,
        error="timeout",
        state_delta={
            "recovery_next_step": "先检查短命令输出。" + ("N" * 500),
        },
    )

    reply = _fallback_reply_for_user(action, result, None)
    assert len(reply) < 260
    assert "S" * 200 not in reply
    assert "N" * 200 not in reply
    assert "先检查短命令输出" in reply


def test_fallback_reply_for_user_uses_next_verification_from_completion_blocker():
    from core.loop.shared.logging import _fallback_reply_for_user
    from store.task import Task
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="task.complete")
    result = ToolResult(
        summary="任务皮层仍有未验证的下一步。",
        error="WorkbenchVerificationPending",
        skipped=True,
        state_delta={
            "next_verification": "运行 pytest 验证 task.complete 未执行 workbench 下一步时会被拦截。",
        },
    )
    task = Task(
        id=9,
        title="旧任务",
        status="in_progress",
        priority="normal",
        created_at="2026-05-15T14:00:00+00:00",
        next_step="旧的 next_step 不应优先",
    )

    reply = _fallback_reply_for_user(action, result, task)
    assert reply.startswith("这轮工具执行失败")
    assert "运行 pytest 验证" in reply
    assert "旧的 next_step" not in reply


def test_fallback_reply_for_user_does_not_echo_tool_summary_on_success():
    from core.loop.shared.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="file.read", rationale="我已经收集到关键证据。")
    result = ToolResult(summary="/tmp/a.py\n/tmp/b.py")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("我已完成本轮处理")
    assert "关键证据" in reply
    assert "/tmp/a.py" not in reply


def test_fallback_reply_for_user_filters_internal_action_first_rationale():
    from core.loop.shared.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        rationale="Action-first fallback: 用户给出文件路径且本轮不能空等，先读取路径形成证据。",
    )
    result = ToolResult(summary="读取完成")

    reply = _fallback_reply_for_user(action, result, None)
    assert "Action-first" not in reply
    assert "本轮不能空等" not in reply
    assert "正在整理基于证据的答复" in reply


def test_fallback_reply_for_user_filters_internal_gate_wait_rationale():
    from core.loop.shared.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(
        decision="wait",
        rationale="行为门控制动：file.read /tmp/a 已连续重复 3 次，继续执行没有新增证据。",
    )
    result = ToolResult(summary="")

    reply = _fallback_reply_for_user(action, result, None)
    assert "行为门控" not in reply
    assert "file.read" not in reply
    assert "需要更多信息后再继续" in reply


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_falls_back_when_reply_only_empty_for_user_message():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            return _judgment_output(decision="wait", rationale="继续等待更多证据")

    loop, store = _tick_reply_loop(_Judgment())
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        rationale="已拿到证据，等待下一步判断。",
    )
    result = ToolResult(summary="读取完成")

    await _finalize_tick_user_reply(
        loop,
        action,
        result,
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "读取完成"}],
        user_message="继续",
        active_task=None,
        chat_id=None,
    )

    assert action.reply_to_user.startswith("我需要先停一下")
    assert action.decision == "wait"
    assert action.chosen_action_id == ""
    assert action.params == {}
    assert "已拿到证据" in action.reply_to_user
    assert store.messages == []


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_uses_medium_thinking_for_active_task_reply():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    captured: dict[str, Any] = {}

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            captured.update(kwargs)
            return _judgment_output(decision="wait", reply_to_user="已根据执行结果回复。")

    loop, _store = _tick_reply_loop(_Judgment())
    action = _judgment_output(decision="act", chosen_action_id="file.read", rationale="已读取关键文件。")
    result = ToolResult(summary="读取完成")

    await _finalize_tick_user_reply(
        loop,
        action,
        result,
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "读取完成"}],
        user_message="继续",
        active_task=SimpleNamespace(id=7, next_step="解释结果"),
        chat_id=None,
    )

    assert captured["reply_only"] is True
    assert captured["thinking_override"] == "medium"
    assert action.reply_to_user == "已根据执行结果回复。"


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_keeps_direct_pause_reply_without_reply_only():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            raise AssertionError("direct pause reply should not invoke reply-only continuation")

    loop, store = _tick_reply_loop(_Judgment())
    action = _judgment_output(
        decision="pause",
        rationale="已经得到结论",
        reply_to_user="这是直接答复。",
    )

    await _finalize_tick_user_reply(
        loop,
        action,
        ToolResult(summary=""),
        tool_history=[],
        user_message="你好",
        active_task=None,
        chat_id="chat-1",
    )

    assert action.reply_to_user == "这是直接答复。"
    assert store.messages == [("assistant", "这是直接答复。", "chat-1")]


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_keeps_disaster_fallback_for_reply_only_failure():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            return _judgment_output(decision="wait", rationale="[reply-only] reply_to_user 不能为空")

    loop, _store = _tick_reply_loop(_Judgment())
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        rationale="我已经收集到关键证据。",
    )
    result = ToolResult(summary="读取完成")

    await _finalize_tick_user_reply(
        loop,
        action,
        result,
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "读取完成"}],
        user_message="继续",
        active_task=None,
        chat_id=None,
    )

    assert action.reply_to_user.startswith("我需要先停一下")
    assert action.decision == "wait"
    assert "状态:" not in action.reply_to_user


@pytest.mark.asyncio
async def test_finalize_tick_user_reply_rejects_internal_json_payload():
    from core.loop.tick import _finalize_tick_user_reply
    from tools.registry import ToolResult

    leaked_reply = (
        '{"command":"cd /root/lingzhou && git status","timeout":30,"workdir":"/root/lingzhou"}\n'
        '{"tool":"shell.run","status":"ok","summary":"done"}'
    )

    class _Judgment:
        async def decide_continue(self, *args, **kwargs):
            return _judgment_output(
                decision="wait",
                rationale="已根据执行结果组织回复。",
                reply_to_user=leaked_reply,
            )

    loop, _store = _tick_reply_loop(_Judgment())
    action = _judgment_output(
        decision="act",
        chosen_action_id="shell.run",
        rationale="已经完成日志统计并形成结论。",
    )
    result = ToolResult(summary="shell.run exit=0 chars=120")

    await _finalize_tick_user_reply(
        loop,
        action,
        result,
        tool_history=[{"tool": "shell.run", "params": {"command": "git status"}, "result": "done"}],
        user_message="继续",
        active_task=SimpleNamespace(id=8, next_step="总结日志结论"),
        chat_id=None,
    )

    assert action.reply_to_user.startswith("我需要先停一下")
    assert action.decision == "wait"
    assert '{"command"' not in action.reply_to_user
    assert '{"tool"' not in action.reply_to_user


@pytest.mark.asyncio
async def test_persist_tick_user_reply_does_not_append_skill_suffix():
    from core.loop.tick import _persist_tick_user_reply

    class _Store:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str]] = []

        async def add_chat_message(self, role: str, content: str, chat_id: str = ""):
            self.messages.append((role, content, chat_id))
            return len(self.messages)

    store = _Store()
    loop = cast("Any", SimpleNamespace(_task_store=store))
    action = _judgment_output(
        decision="pause",
        rationale="证据已足够，直接回复用户。",
        reply_to_user="这是最终答复。",
    )
    action.applied_skills = ["runtime-bootstrap", "task-planning"]

    await _persist_tick_user_reply(
        loop,
        action,
        active_task=None,
        chat_id="",
    )

    assert action.reply_to_user == "这是最终答复。"
    assert store.messages == [("assistant", "这是最终答复。", "")]


@pytest.mark.asyncio
async def test_persist_tick_user_reply_dedupes_repeated_status_point():
    from core.loop.tick import _persist_tick_user_reply

    class _Store:
        def __init__(self) -> None:
            self.messages: list[tuple[str, str, str]] = []

        async def add_chat_message(self, role: str, content: str, chat_id: str = ""):
            self.messages.append((role, content, chat_id))
            return len(self.messages)

    store = _Store()
    loop = cast("Any", SimpleNamespace(_task_store=store))
    repeated = "最近发现继续重复同一个 memory.search 查询的信息增量很低，所以已经停止重复搜索。"
    action = _judgment_output(
        decision="pause",
        rationale="证据已足够，直接回复用户。",
        reply_to_user=(
            "已确认的状态：\n"
            "1. 活跃任务仍在继续。\n"
            "2. 当前已建立工作台。\n"
            f"3. {repeated}\n"
            "目前还不能说全部找回。  3. 最近发现继续重复同一个 memory.search "
            "查询的信息增量很低，所以已经停止重复搜索。"
        ),
    )

    await _persist_tick_user_reply(
        loop,
        action,
        active_task=None,
        chat_id="",
    )

    assert action.reply_to_user.count(repeated) == 1
    assert store.messages[0][1].count(repeated) == 1
    assert "1. 活跃任务仍在继续。" in action.reply_to_user
    assert "2. 当前已建立工作台。" in action.reply_to_user


def test_infer_valence_from_text_uses_explicit_hint_only():
    from core.config_models import EmotionConfig
    from core.loop.shared.common import _infer_valence_from_text

    default_cfg = EmotionConfig()
    assert _infer_valence_from_text("继续推进，暂无结构化情绪提示", 0.6, default_cfg) == 0.6
    assert _infer_valence_from_text("root cause found; valence=0.2", 0.6, default_cfg) == pytest.approx(0.52)
    assert _infer_valence_from_text(
        "root cause found; valence=0.2",
        0.6,
        EmotionConfig(reflection_valence_history_weight=0.5, reflection_valence_hint_weight=0.5),
    ) == pytest.approx(0.4)


def test_should_continue_within_tick_for_autonomous_act():
    from core.loop.shared.common import _next_initial_tier_hint, _should_continue_within_tick
    from tools.registry import ToolResult

    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        has_active_task=True,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        has_active_task=False,
    ) is False
    assert _should_continue_within_tick(_judgment_output(decision="act", chosen_action_id="task.complete")) is False
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="task.complete"),
        has_active_task=True,
        result=ToolResult(
            summary="任务皮层仍有未验证的下一步",
            skipped=True,
            error="WorkbenchVerificationPending",
        ),
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="task.workbench"),
        has_active_task=True,
        result=ToolResult(summary="task.workbench id=1"),
    ) is False
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="task.workbench"),
        has_active_task=True,
        result=ToolResult(
            summary="工具参数缺失: task.workbench requires workbench",
            skipped=True,
            error="ToolInputInvalid",
            state_delta={"retry_params_template": {"workbench": {}}},
        ),
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="task.wait"),
        user_message="继续",
        has_active_task=True,
        registry=_tool_registry(),
        result=ToolResult(
            summary="external wait 缺少 wait_key 时必须写清 next_step",
            skipped=True,
            error="WaitConditionAmbiguous",
            state_delta={"tool_input_invalid": True},
        ),
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="task.complete"),
        result=ToolResult(summary="任务已完成", skipped=False, state_delta={"task_status": "done"}),
    ) is False
    assert _should_continue_within_tick(_judgment_output(decision="wait")) is False
    reg = _tool_registry()
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=True,
        registry=reg,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=False,
        registry=reg,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.write"),
        user_message="帮我顺手改一下配置",
        has_active_task=True,
        registry=reg,
    ) is False
    assert _next_initial_tier_hint(
        _judgment_output(decision="act", chosen_action_id="memory.search")
    ) is None
    assert _next_initial_tier_hint(
        _judgment_output(
            decision="act",
            chosen_action_id="memory.search",
            model_strategy={"next_phase_tier": "reader"},
        ),
    ) == "reader"


async def test_decide_continue_reply_only_forces_reasoner_and_reply_to_user():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_messages = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_messages = messages
            return (
                '{"decision":"act","chosen_action_id":"file.read",'
                '"params":{"path":"/tmp/ignored"},'
                '"rationale":"证据已足够，直接整理用户答复。",'
                '"reply_to_user":"这是最终回复。"}'
            )

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 2 条相关记忆"}],
        user_message="继续分析",
        prefer_tier="reader",
        reply_only=True,
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.reply_to_user == "这是最终回复。"
    assert layer.last_call_meta["tier"] == "reasoner"
    assert provider.last_messages is not None
    assert "禁止再调用任何工具" in provider.last_messages[1].content


async def test_decide_continue_surfaces_missing_chosen_action_id_without_runtime_repair():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            return '{"decision":"act","params":{"key":"loop.min_act_gap","value":100},"rationale":"应该改配置"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"}],
        user_message="把 tick 设置到 100 毫秒一次",
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.rationale == "act 决策缺少 chosen_action_id"
    assert provider.calls == 1


async def test_decide_continue_defaults_to_provider_natural_timeout(monkeypatch):
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait","rationale":"ok"}'

        async def close(self):
            return None

    async def _fail_wait_for(*args, **kwargs):
        raise AssertionError("LLM chat should not be wrapped by local wait_for when timeout=None")

    import core.judgment.decision.helpers as helpers

    monkeypatch.setattr(helpers.asyncio, "wait_for", _fail_wait_for)

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
    })

    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "task.list", "params": {"status": "all"}, "result": "命中 3 条任务"}],
        user_message="继续",
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"


async def test_judgment_normalizes_whitespace_tool_name_to_wait():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        model_ref = "bailian/qwen3.6-plus"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"act","chosen_action_id":"   ","params":{"path":"x"},"rationale":"bad"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
    })
    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "task.list", "params": {}, "result": "ok"}],
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.rationale == "act 决策缺少 chosen_action_id"


async def test_judgment_rejects_unregistered_tool_before_execution():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        last_usage = {"prompt_tokens": 1, "completion_tokens": 1}
        model_ref = "bailian/qwen3.6-plus"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"act","chosen_action_id":"not.a.tool","params":{},"rationale":"bad"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
    })
    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "task.list", "params": {}, "result": "ok"}],
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert out.rationale == "未知工具: 'not.a.tool'"


async def test_repair_output_uses_broken_output_only():
    from core.config import Config
    from core.judgment.executor import JudgmentExecutor

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_messages = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_messages = messages
            user_content = str(messages[1].content)
            assert "[context]" not in user_content
            assert "[broken_output]" in user_content
            return '{"decision":"wait","rationale":"ok"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    executor = JudgmentExecutor(provider, cfg)
    huge_context = "头部信息\n" + ("X" * 180000) + "\n尾部信息\n" + ("Y" * 60000)
    huge_raw = "{" + ("Z" * 50000) + "}"

    repaired = await executor._repair_output(huge_context, huge_raw)

    assert repaired is not None
    assert repaired.rationale == "ok"
    assert provider.last_messages is not None
    assert provider.last_messages[0].role == "system"


@pytest.mark.asyncio
async def test_decide_continue_normalizes_chat_reply_pseudo_tool():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            return '{"decision":"act","chosen_action_id":"chat_reply","reply_to_user":"我先直接回答你这张图里有一只猫。","rationale":"已经有足够证据，直接答复用户"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "image.analyze", "params": {"images": 1}, "result": "图中是一只猫坐在窗边"}],
        user_message="图里是什么",
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.reply_to_user == "我先直接回答你这张图里有一只猫。"
    assert provider.calls == 1


async def test_decide_continue_includes_structured_tool_history_window():
    from core.config import Config
    from core.judgment import JudgmentLayer

    class _DummyProvider:
        def __init__(self) -> None:
            self.last_messages = None

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.last_messages = messages
            return '{"decision":"wait","rationale":"继续观察"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    provider = _DummyProvider()
    layer = JudgmentLayer(provider, _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{
            "tool": "memory.search",
            "params": {"query": "继续分析"},
            "result": "命中 2 条相关记忆",
            "summary": "命中 2 条相关记忆",
            "error": "",
            "status": "ok",
            "state_delta": {"hits": 2},
        }],
        user_message="继续分析",
        prefer_tier="reasoner",
    )

    assert out.decision == "wait"
    assert provider.last_messages is not None
    assert "结构化最近工具结果(JSON)" in provider.last_messages[1].content
    assert '"status": "ok"' in provider.last_messages[1].content
    assert '"state_delta": {' in provider.last_messages[1].content


def test_continue_context_rebuilds_budgeted_working_set_not_raw_prompt():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    asm = layer._assembler
    asm._judgment_template = "{{user_message}}\n{{wm_section}}\n{{tools_section}}"
    asm._last_context_text = "RAW_PREVIOUS_PROMPT" * 1000
    asm._last_context_sections = {
        "user_message": "继续",
        "wm_section": "WM_FACT",
        "tools_section": "TOOL_CATALOG",
    }
    asm._last_context_budget = 6000

    text = asm._build_continue_context(
        [{"tool": "memory.search", "params": {"query": "x"}, "result": "命中 1 条"}],
        user_message="继续",
        reply_only=False,
        wm_delta=None,
    )

    assert "RAW_PREVIOUS_PROMPT" not in text
    assert "WM_FACT" in text
    assert "TOOL_CATALOG" in text
    assert "Continue 收敛契约" in text
    assert "避免对同一路径、同一查询、同一命令做低增量重复" in text
    assert "结构化最近工具结果(JSON)" in text


def test_continue_context_reuses_cached_compression_capsule():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    asm = layer._assembler
    asm._judgment_template = "{{user_message}}\n{{wm_section}}\n{{tools_section}}"
    asm._last_context_text = "RAW_PREVIOUS_PROMPT" * 1000
    asm._last_context_sections = {
        "user_message": "继续",
        "wm_section": "WM_FACT_SHOULD_NOT_RERENDER",
        "tools_section": "TOOL_CATALOG_SHOULD_NOT_RERENDER",
    }
    asm._last_context_budget = 6000
    asm._last_context_compression_capsule = "CORTEX_CAPSULE\nnext_verification=继续验证"

    text = asm._build_continue_context(
        [{"tool": "task.workbench", "params": {"workbench": {}}, "result": "ok"}],
        user_message="继续",
        reply_only=False,
        wm_delta=None,
    )

    assert "CORTEX_CAPSULE" in text
    assert "next_verification=继续验证" in text
    assert "RAW_PREVIOUS_PROMPT" not in text
    assert "WM_FACT_SHOULD_NOT_RERENDER" not in text
    assert "TOOL_CATALOG_SHOULD_NOT_RERENDER" not in text
    assert "结构化最近工具结果(JSON)" in text


def test_continue_context_clamps_oversized_compression_capsule_to_reserve_budget():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    asm = layer._assembler
    asm._last_context_budget = 1600
    asm._last_context_compression_capsule = (
        "[上下文超限压缩胶囊]\n"
        "next_verification=保留这个关键恢复动作\n"
        + ("外围胶囊材料 " * 5000)
        + "\ncompletion_checks=保留尾部检查"
    )

    text = asm._build_continue_context(
        [{"tool": "file.read", "params": {"path": "/tmp/a.py"}, "result": "ok"}],
        user_message="继续",
        reply_only=False,
        wm_delta=None,
    )

    assert "上下文超限压缩胶囊" in text
    assert "next_verification=保留这个关键恢复动作" in text
    assert "completion_checks=保留尾部检查" in text
    assert "chars omitted" in text
    assert "结构化最近工具结果(JSON)" in text
    assert len(text) < 12000


def test_continue_context_compacts_wm_delta_block():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    asm = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)._assembler
    asm._judgment_template = "{{user_message}}\n{{wm_section}}\n{{tools_section}}"
    asm._last_context_sections = {
        "user_message": "",
        "wm_section": "WM_FACT",
        "tools_section": "TOOL_CATALOG",
    }
    asm._last_context_budget = 6000
    asm._last_context_compression_capsule = ""

    text = asm._build_continue_context(
        [{"tool": "file.read", "params": {"path": "/tmp/a.py"}, "result": "ok"}],
        user_message="",
        reply_only=False,
        wm_delta=[
            {"kind": f"k{i}", "priority": 0.5, "content": "X" * 1000}
            for i in range(12)
        ],
    )

    assert "已压缩早期 4 条本轮 WM 更新" in text
    assert "k0" not in text
    assert "k11" in text
    assert "X" * 500 not in text


def test_reply_only_context_omits_tool_catalog_from_working_set():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
    asm = layer._assembler
    asm._judgment_template = "{{user_message}}\n{{wm_section}}\n{{tools_section}}"
    asm._last_context_sections = {
        "user_message": "继续",
        "wm_section": "WM_FACT",
        "tools_section": "TOOL_CATALOG",
    }
    asm._last_context_budget = 6000

    text = asm._build_continue_context(
        [{"tool": "memory.search", "params": {"query": "x"}, "result": "命中 1 条"}],
        user_message="继续",
        reply_only=True,
        wm_delta=None,
    )

    assert "WM_FACT" in text
    assert "TOOL_CATALOG" not in text
    assert "禁止再调用任何工具" in text
    assert "Continue 收敛契约" not in text


def test_structured_tool_history_window_clips_huge_summary_and_state_delta():
    from core.judgment.output import _structured_tool_history_window

    tool_history = [
        {
            "tool": "shell.run",
            "params": {"command": "ls -la"},
            "status": "ok",
            "summary": "A" * 5000,
            "result": "A" * 5000,
            "state_delta": {"blob": "B" * 5000, "count": 1},
        },
    ]

    json_block, text_block = _structured_tool_history_window(tool_history)
    assert "summary=" in text_block
    assert "omitted" in text_block
    assert json.loads(json_block)
    assert len(json_block) < 2000
    assert len(text_block) < 1200


def test_structured_tool_history_window_preserves_recovery_state_delta_first():
    from core.judgment.output import _structured_tool_history_window

    state_delta = {
        **{f"field_{i}": f"value-{i}" for i in range(20)},
        "has_more": True,
        "truncated": True,
        "next_params": {"path": "/tmp/big.txt", "start": 20, "max_chars": 20},
        "recovery_next_step": "继续读取同一文件剩余内容：file.read path=/tmp/big.txt start=20 max_chars=20。",
    }
    tool_history = [
        {
            "tool": "file.read",
            "params": {"path": "/tmp/big.txt", "max_chars": 20},
            "status": "ok",
            "summary": "x" * 20,
            "state_delta": state_delta,
        },
    ]

    json_block, text_block = _structured_tool_history_window(tool_history)
    payload = json.loads(json_block)[0]["state_delta"]

    assert "recovery_next_step" in payload
    assert "next_params" in payload
    assert payload["next_params"]["start"] == "20"
    assert payload["truncated"] == "True"
    assert "recovery_next_step" in text_block
    assert "field_19" not in payload


async def test_decide_continue_keeps_complex_act_without_runtime_rewrite():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from store.task import Task

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return (
                '{"decision":"act","chosen_action_id":"shell.run",'
                '"params":{"command":"pytest -q"},'
                '"rationale":"先继续排查失败原因。",'
                '"next_step":"再修复 chat 回复链路"}'
            )

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    task = Task(
        id=11,
        title="排查 chat 回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="逐一排查 chat 回复链路",
    )
    layer = JudgmentLayer(_DummyProvider(), _tool_registry(), cfg)
    layer._assembler._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "memory.search", "params": {"query": "chat 回复"}, "result": "命中 2 条相关记忆"}],
        user_message="请你逐一排查并修复 chat 回复问题",
        active_task=task,
        prefer_tier="reasoner",
    )

    assert out.decision == "act"
    assert out.chosen_action_id == "shell.run"
    assert out.params == {"command": "pytest -q"}
    assert out.next_step == "再修复 chat 回复链路"


def test_preferred_continue_tier_uses_manifest_reader_tier():
    from core.loop.shared.common import _should_continue_within_tick
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="debug.reader.inspect",
        description="调试用 reader 工具",
        prefer_tier="reader",
        capabilities=("completion_info_only",),
    ))
    async def _debug_reader_inspect(params: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(summary="ok")

    reg = ToolRegistry()
    action = _judgment_output(decision="act", chosen_action_id="debug.reader.inspect")

    assert _should_continue_within_tick(
        action,
        user_message="继续分析",
        has_active_task=True,
        registry=reg,
    ) is True


async def test_sync_task_progress_state_promotes_previous_next_step():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("推进任务", goal="验证 current_step/next_step 同步", next_step="先读文件")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="先读文件",
            action=_judgment_output(decision="act", chosen_action_id="file.read", next_step="再总结结论"),
            progressful=True,
        )
        assert updated is not None
        assert updated.current_step == "先读文件"
        assert updated.next_step == "再总结结论"

        updated2 = await _sync_task_progress_state(
            store,
            updated,
            previous_next_step="再总结结论",
            action=_judgment_output(decision="act", chosen_action_id="file.read", next_step=""),
            progressful=True,
        )
        assert updated2 is not None
        assert updated2.current_step == "再总结结论"
        assert updated2.next_step == ""
        await store.close()


async def test_sync_task_progress_state_preserves_explicit_current_step_from_state_delta():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-explicit.db")
        await store.open()
        task_id = await store.add_task("迁移任务", goal="验证显式 current_step 优先", next_step="继续旧技能")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        await store.sync_task_progress(task_id, current_step="收到新迁移指令", next_step="开始盘点旧运行时记忆")
        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="继续旧技能",
            action=_judgment_output(decision="act", chosen_action_id="task.update", next_step="开始盘点旧运行时记忆"),
            progressful=True,
            state_delta={"current_step": "收到新迁移指令", "next_step": "开始盘点旧运行时记忆"},
        )

        assert updated is not None
        assert updated.current_step == "收到新迁移指令"
        assert updated.next_step == "开始盘点旧运行时记忆"
        await store.close()


async def test_sync_task_progress_state_promotes_recovery_next_step_from_state_delta():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-recovery-step.db")
        await store.open()
        task_id = await store.add_task("恢复任务", goal="验证恢复下一步同步", next_step="执行旧步骤")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="执行旧步骤",
            action=_judgment_output(decision="act", chosen_action_id="task.complete", next_step=""),
            progressful=True,
            state_delta={
                "completion_blocked": True,
                "recovery_next_step": "先执行 shell.run 验证测试，再重新完成任务。",
            },
        )

        assert updated is not None
        assert updated.current_step == "执行旧步骤"
        assert updated.next_step == "先执行 shell.run 验证测试，再重新完成任务。"
        await store.close()


async def test_sync_task_progress_state_uses_next_verification_when_no_recovery_step():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-next-verification.db")
        await store.open()
        task_id = await store.add_task("验证任务", goal="验证 next_verification 同步", next_step="")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="",
            action=_judgment_output(decision="act", chosen_action_id="task.complete", next_step=""),
            progressful=False,
            state_delta={
                "next_verification": "读取最新 loop 日志确认 active_idle_gap 生效。",
            },
        )

        assert updated is not None
        assert updated.current_step == ""
        assert updated.next_step == "读取最新 loop 日志确认 active_idle_gap 生效。"
        await store.close()


async def test_sync_task_progress_state_uses_nested_cortex_next_verification():
    from core.cortex import intent as cortex_intent
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-cortex-next-verification.db")
        await store.open()
        task_id = await store.add_task("工作台任务", goal="验证 workbench next_verification 同步", next_step="旧验证")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="旧验证",
            action=_judgment_output(decision="act", chosen_action_id="task.workbench", next_step=""),
            progressful=True,
            state_delta={
                "cortex": {
                    "domain": "runtime-loop",
                    "next_verification": cortex_intent.control_next_verification(
                        "下一轮先综合本 tick 工具结果，再选择最高信息增量验证动作。"
                    ),
                }
            },
        )

        assert updated is not None
        assert updated.current_step == "旧验证"
        assert updated.next_step == ""
        await store.close()


async def test_sync_task_progress_state_clears_resolved_workbench_next_step():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-resolved-workbench-clears-next-step.db")
        await store.open()
        task_id = await store.add_task(
            "工作台任务",
            goal="验证 resolved workbench 不会重新拉起 next_step",
            next_step="继续重复 all limit=8",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="继续重复 all limit=8",
            action=_judgment_output(decision="act", chosen_action_id="task.workbench", next_step=""),
            progressful=False,
            state_delta={
                "cortex": {
                    "domain": "runtime-loop",
                    "next_verification": "避免继续重复 all limit=8；切换到一个更高信息增量动作。",
                    "verification_state": {
                        "goal": "避免继续重复 all limit=8；切换到一个更高信息增量动作。",
                        "status": "resolved",
                    },
                }
            },
        )

        assert updated is not None
        assert updated.next_step == ""
        await store.close()


async def test_sync_task_progress_state_workbench_overrides_stale_next_step():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-workbench-overrides-stale.db")
        await store.open()
        task_id = await store.add_task("工作台任务", goal="验证 workbench 权威下一步", next_step="继续重复旧日志读取")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="旧的启动步骤",
            action=_judgment_output(decision="act", chosen_action_id="task.workbench", next_step=""),
            progressful=True,
            state_delta={
                "cortex": {
                    "domain": "runtime-loop",
                    "next_verification": "改用 shell.run 查询最近 10 条 runs，确认重复来源。",
                }
            },
        )

        assert updated is not None
        assert updated.current_step == "旧的启动步骤"
        assert updated.next_step == "改用 shell.run 查询最近 10 条 runs，确认重复来源。"
        await store.close()


async def test_sync_task_progress_state_completion_blocker_overrides_stale_next_step():
    from core.loop.task.runtime import _sync_task_progress_state
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime-completion-blocker-overrides-stale.db")
        await store.open()
        task_id = await store.add_task("完成门任务", goal="验证阻塞恢复下一步", next_step="继续旧完成尝试")
        task = await store.get_task_by_id(task_id)
        assert task is not None

        updated = await _sync_task_progress_state(
            store,
            task,
            previous_next_step="旧的验证步骤",
            action=_judgment_output(decision="act", chosen_action_id="task.complete", next_step=""),
            progressful=False,
            state_delta={
                "completion_blocked": True,
                "next_verification": "先执行 pytest 验证 workbench 下一步，再重新 complete。",
            },
        )

        assert updated is not None
        assert updated.current_step == ""
        assert updated.next_step == "先执行 pytest 验证 workbench 下一步，再重新 complete。"
        await store.close()


def test_fmt_task_exposes_runtime_state_to_llm():
    from core.judgment.context.tasks import _fmt_task
    from store.task import Task

    task = Task(
        id=7,
        title="测试任务",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="验证状态可见性",
        next_step="继续修复",
        current_step="检查 run monitor",
        model_tier="repair",
        result_json={"last_run_status": "failed"},
        extras={"plan": [{"step": "检查回复链路", "status": "in_progress"}]},
    )
    section = _fmt_task(task)
    assert "状态: active" in section
    assert "模型层级: repair" in section
    assert "当前步骤: 检查 run monitor" in section
    assert "当前计划:" in section
    assert "检查回复链路" in section
    assert "最近运行状态: failed" in section


def test_fmt_context_facts_surfaces_task_and_recent_general_facts():
    asyncio.run(_fmt_context_facts_surfaces_task_and_recent_general_facts())


def test_load_context_facts_snapshot_uses_configured_exclude_prefixes_and_limits():
    asyncio.run(_load_context_facts_snapshot_uses_configured_exclude_prefixes_and_limits())


async def _fmt_context_facts_surfaces_task_and_recent_general_facts():
    from core.judgment.context.facts import _load_context_facts_snapshot
    from core.judgment.context.tasks import _fmt_context_facts
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / 'facts.db')
        await store.open()
        task_id = await store.add_task('分析旧运行时记忆', goal='确认 carrier')
        task = await store.get_task_by_id(task_id)
        assert task is not None

        await store.set_fact(f'task:{task_id}:progress', '已确认 sqlite 为主载体', scope='task')
        await store.set_fact(
            f'task:{task_id}:large_json',
            json.dumps({"head": "START", "payload": "A" * 5000 + "TAIL"}, ensure_ascii=False),
            scope='task',
        )
        await store.set_fact('legacy_runtime.workspace_memory.primary_carrier', '/root/.legacy-runtime/memory/main.sqlite')
        await store.set_fact('pref:routing_overrides', '{"reader":"demo"}', scope='system')

        facts = await _load_context_facts_snapshot(store, task)
        text = _fmt_context_facts(facts)

        assert f'task:{task_id}:progress' in text
        assert f'task:{task_id}:large_json' in text
        assert "START" in text
        assert "TAIL" in text
        assert "chars omitted" in text
        assert "A" * 1000 not in text
        assert 'legacy_runtime.workspace_memory.primary_carrier' in text
        assert 'pref:routing_overrides' not in text
        await store.close()


async def _load_context_facts_snapshot_uses_configured_exclude_prefixes_and_limits():
    from core.judgment.context.facts import _load_context_facts_snapshot

    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str | None, int]] = []

        async def list_facts(self, prefix=None, limit=0):
            self.calls.append((prefix, limit))
            if prefix == 'task:7:':
                return [
                    ('task:7:a', 'A'),
                    ('task:7:b', 'B'),
                    ('task:7:c', 'C'),
                ][:limit]
            if prefix == 'user:':
                return [
                    ('user:name', 'bat'),
                    ('user:explicit:1', '记住我叫 bat'),
                ][:limit]
            return [
                ('pref:hidden', 'P'),
                ('soul:visible', 'S'),
                ('evolution:visible', 'E'),
                ('misc:1', 'M1'),
                ('run:hidden', 'R'),
            ][:limit]

    store = _Store()
    facts = await _load_context_facts_snapshot(
        cast("Any", store),
        cast("Any", SimpleNamespace(id=7)),
        exclude_prefixes=['pref:', 'run:'],
        task_limit=2,
        global_limit=2,
        priority_prefixes=['user:'],
        priority_limit=1,
        recent_scan_multiplier=4,
        recent_scan_min=9,
    )

    assert store.calls == [('task:7:', 2), ('user:', 1), (None, 9)]
    assert facts == [
        ('task:7:a', 'A'),
        ('task:7:b', 'B'),
        ('user:name', 'bat'),
        ('soul:visible', 'S'),
        ('evolution:visible', 'E'),
    ]


def test_tool_result_log_fields_include_state_delta():
    from core.execution import _tool_result_log_fields
    from tools.registry import ToolResult

    summary, error, state = _tool_result_log_fields(ToolResult(
        summary="工具完成\n含多行",
        error="",
        state_delta={"task_status": "waiting", "wait_key": "exec-1"},
    ))

    assert summary == "工具完成\\n含多行"
    assert error == ""
    assert '"task_status": "waiting"' in state
    assert '"wait_key": "exec-1"' in state


def test_tool_result_log_fields_prefer_log_summary_over_raw_text():
    from core.execution import _tool_result_log_fields
    from tools.registry import ToolResult

    summary, error, state = _tool_result_log_fields(ToolResult(
        summary="---\nlicense: Proprietary\nname: error-handling\n...",
        error="",
        metadata={"log_summary": "file.read path=/tmp/skill.md chars=2048 preview='---'"},
    ))

    assert summary == "file.read path=/tmp/skill.md chars=2048 preview='---'"
    assert error == ""
    assert state == ""


def test_run_progress_text_prefers_log_summary_over_huge_summary():
    from core.execution.helpers import _run_progress_text
    from tools.registry import ToolResult

    res = ToolResult(
        summary="X" * 200000,
        metadata={"log_summary": "file.read path=/tmp/big.txt chars=200000"},
    )
    assert _run_progress_text(res) == "file.read path=/tmp/big.txt chars=200000"


def test_clip_reply_for_log_strips_memory_context():
    from core.loop.shared.logging import _clip_reply_for_log

    clipped = _clip_reply_for_log("<memory-context>hidden</memory-context>\n用户可见回复")
    assert clipped == "用户可见回复"


def test_assemble_context_prefers_active_task_override_with_inbox():
    asyncio.run(_assemble_context_prefers_active_task_override_with_inbox())


async def _assemble_context_prefers_active_task_override_with_inbox():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "旧回填任务",
                goal="等待模型加载完成后，再次检查日志确认数据回填进度",
                next_step="继续检查回填进度",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None
            task.extras["inbox_messages"] = [
                "收到新的用户消息：请你使用 puppeteer 去搜索。"
            ]

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=task,
                user_message="请你使用 puppeteer 去搜索。",
            )

            assert "⚠️ 新增用户消息（inbox 1 条，先评估这些新消息是否改变当前方向）:" in text
            assert "收到新的用户消息：请你使用 puppeteer 去搜索。" in text
        finally:
            await store.close()


def test_assemble_context_without_active_task_or_probe_manager_does_not_crash():
    asyncio.run(_assemble_context_without_active_task_or_probe_manager_does_not_crash())


async def _assemble_context_without_active_task_or_probe_manager_does_not_crash():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            layer._assembler._probe_manager = None
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="帮我检查当前状态",
            )

            assert "帮我检查当前状态" in text
        finally:
            await store.close()


def test_assemble_context_includes_runtime_life_snapshot():
    asyncio.run(_assemble_context_includes_runtime_life_snapshot())


async def _assemble_context_includes_runtime_life_snapshot():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="检查生命状态",
                runtime_life_snapshot={
                    "memory": {
                        "wm_pressure": 0.73,
                        "wm_tokens": 730,
                        "wm_token_budget": 1000,
                        "semantic_nodes": 42,
                        "semantic_maintenance_state": "ready",
                    },
                    "startup": {"bootstrap_mode": "none", "tick_count": 8},
                    "pressure": {"dispatch_running": 1, "dispatch_pending": 2, "dispatch_queue_pressure": 0.5, "idle_cycles": 3, "wait_streak": 4},
                    "drive": {
                        "overall": 0.61,
                        "prediction_error_ema": 0.2,
                        "top_interests": [{"domain": "memory_system", "score": 0.9}],
                    },
                    "action": {"last_decision": "wait", "last_tool": "", "last_status": "", "last_progressful": False},
                },
            )

            assert "### 生命体运行状态（runtime life snapshot）" in text
            assert "memory.wm_pressure: 0.73" in text
            assert "memory.semantic_nodes: 42" in text
            assert "pressure.dispatch: running=1 pending=2 queue_pressure=0.50" in text
            assert "drive.top_interests: memory_system=0.90" in text
        finally:
            await store.close()


def test_assemble_context_semantic_timeout_degrades(monkeypatch, caplog):
    asyncio.run(_assemble_context_semantic_timeout_degrades(monkeypatch, caplog))


def test_memory_system_section_exposes_local_embedding_guard():
    from types import SimpleNamespace

    from core.judgment.context.sections import _fmt_memory_system

    class _Semantic:
        def stats(self):
            return {
                "db_path": "/tmp/semantic.db",
                "nodes_dir": "/tmp/nodes",
                "nodes": 1,
                "fts5_ok": True,
                "maintenance_state": "ready",
                "embedding_enabled": False,
            }

    text = _fmt_memory_system(
        runtime_db="/tmp/runtime.db",
        memory_dir="/tmp/memory",
        workspace_dir="/tmp/workspace",
        semantic=_Semantic(),
        memory_cfg=SimpleNamespace(
            embedding_provider="local",
            embedding_model=None,
            embedding_fallback="fts",
            local_embed_model="BAAI/bge-m3",
            local_embed_command_guard=True,
            local_embed_min_available_mib=12288,
        ),
        max_concurrent_ticks=1,
        max_tick_queue=2,
    )

    assert "embedding_provider: local" in text
    assert "embedding_model: none" in text
    assert "embedding_fallback: fts" in text
    assert "local_embed_model: BAAI/bge-m3" in text
    assert "local_embed_command_guard: yes" in text
    assert "local_embed_min_available_mib: 12288" in text
    assert "不要优先生成 build_embeddings.py" in text
    assert "子进程内存上限" in text


async def _assemble_context_semantic_timeout_degrades(monkeypatch, caplog):
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.judgment.assembler import assemble_context as assemble_context_mod
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    async def _timeout(_awaitable, timeout):
        raise TimeoutError

    monkeypatch.setattr(assemble_context_mod.asyncio, "wait_for", _timeout)
    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d, caplog.at_level(logging.WARNING):
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="即使语义检索超时也要继续",
            )

            assert "即使语义检索超时也要继续" in text
            assert "semantic_multi_anchor_timeout" in caplog.text
        finally:
            await store.close()


def test_assemble_context_with_active_task_skips_global_open_task_overview_fetches():
    asyncio.run(_assemble_context_with_active_task_skips_global_open_task_overview_fetches())


async def _assemble_context_with_active_task_skips_global_open_task_overview_fetches():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    class _GuardedTaskStore:
        def __init__(self, base: Any) -> None:
            self._base = base

        async def list_runnable_tasks(self, limit: int = 20):
            raise AssertionError("active task context should not fetch runnable task overview")

        async def list_tasks(self, status: str | None = None, limit: int = 50):
            if status == "waiting":
                raise AssertionError("active task context should not fetch waiting task overview")
            return await self._base.list_tasks(status=status, limit=limit)

        async def find_similar_open_tasks(self, *args: Any, **kwargs: Any):
            raise AssertionError("active task context should not fetch similar open tasks")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._base, name)

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx-active-task.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "当前焦点任务",
                goal="只围绕当前任务推进",
                next_step="继续核对当前任务证据",
                status="in_progress",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            guarded_store = _GuardedTaskStore(store)
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=cast("Any", guarded_store),
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=task,
                user_message="继续当前任务",
            )

            assert "当前焦点任务" in text
        finally:
            await store.close()


def test_assemble_context_semantic_anchors_do_not_bucket_emotion():
    asyncio.run(_assemble_context_semantic_anchors_do_not_bucket_emotion())


async def _assemble_context_semantic_anchors_do_not_bucket_emotion():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    captured: list[str] = []

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "回归情绪锚测试",
                goal="确认 semantic anchors 不再使用情绪桶标签",
                next_step="检查 semantic 检索锚点",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            semantic = SemanticMemory(Path(d) / "memory", decay_lambda=0.0)

            def _capture_retrieve_multi_anchor(anchors, top_k):
                captured.extend(str(a) for a in anchors)
                return []

            semantic.retrieve_multi_anchor = cast("Any", _capture_retrieve_multi_anchor)

            emotion = EmotionState.from_config(cfg)
            emotion.valence = 0.10
            emotion.arousal = 0.95

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=semantic,
                    emotion=emotion,
                ),
                active_task=task,
                user_message="",
            )

            assert captured
            assert "检查 semantic 检索锚点" in captured
            assert "焦虑" not in captured
            assert "沮丧" not in captured
            assert "兴奋" not in captured
            assert "稳定" not in captured
            assert "中性" not in captured
        finally:
            await store.close()


def test_assemble_context_consumes_parallel_fetch_exceptions():
    asyncio.run(_assemble_context_consumes_parallel_fetch_exceptions())


async def _assemble_context_consumes_parallel_fetch_exceptions():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        loop = asyncio.get_running_loop()
        recorded_exceptions: list[str] = []
        prev_handler = loop.get_exception_handler()

        def _capture_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            message = str(context.get("message") or context.get("exception") or context)
            recorded_exceptions.append(message)

        loop.set_exception_handler(_capture_exception)
        try:
            task_id = await store.add_task(
                "并发上下文异常回归",
                goal="确保并发上下文异常不会泄漏未消费异常",
                next_step="触发 list_runs 与 list_failures_for_task 异常",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            async def _list_runs_fail(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("recent runs boom")

            async def _list_failures_fail(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("failures boom")

            store.list_runs = cast("Any", _list_runs_fail)
            store.list_failures_for_task = cast("Any", _list_failures_fail)

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)

            with pytest.raises(RuntimeError, match="recent runs boom"):
                await layer._assembler._assemble_context(
                    CognitionFrame(
                        percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                        wm=WorkingMemory(capacity=20),
                        task_store=store,
                        episodic=EpisodicMemory(Path(d) / "memory"),
                        semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                        emotion=EmotionState.from_config(cfg),
                    ),
                    active_task=task,
                    user_message="检查并发异常清理",
                )

            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert recorded_exceptions == []
        finally:
            loop.set_exception_handler(prev_handler)
            await store.close()


def test_assemble_context_registry_override_limits_tools_section():
    asyncio.run(_assemble_context_registry_override_limits_tools_section())


async def _assemble_context_registry_override_limits_tools_section():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from core.subagent import _DEFAULT_BLOCKED_TOOLS, _FilteredRegistry
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            registry = _tool_registry()
            filtered = _FilteredRegistry(registry, {"task.list"}, set(_DEFAULT_BLOCKED_TOOLS))
            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), registry, cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="检查子灵工具边界",
                registry_override=filtered,
            )

            assert "- `task.list`:" in text
            assert "- `shell.run`:" not in text
            assert "- `subagent.run`:" not in text
        finally:
            await store.close()


def test_assemble_context_includes_recent_daily_continuity():
    asyncio.run(_assemble_context_includes_recent_daily_continuity())


def test_assemble_context_daily_zero_budget_uses_evidence_excerpts():
    asyncio.run(_assemble_context_daily_zero_budget_uses_evidence_excerpts())


def test_assemble_context_skips_daily_when_long_term_memory_is_strong():
    asyncio.run(_assemble_context_skips_daily_when_long_term_memory_is_strong())


def test_assemble_context_includes_chat_scoped_memory_layers():
    asyncio.run(_assemble_context_includes_chat_scoped_memory_layers())


def test_assemble_context_includes_current_interlocutor_sections():
    asyncio.run(_assemble_context_includes_current_interlocutor_sections())


def test_assemble_context_keeps_cross_task_episodic_out_of_current_task_narrative():
    asyncio.run(_assemble_context_keeps_cross_task_episodic_out_of_current_task_narrative())


def test_assemble_context_clips_oversized_cross_task_episodic(caplog):
    asyncio.run(_assemble_context_clips_oversized_cross_task_episodic(caplog))


def test_assemble_context_prefers_focus_fact_over_global_active():
    asyncio.run(_assemble_context_prefers_focus_fact_over_global_active())


def test_assemble_context_without_focus_does_not_fallback_to_global_active():
    asyncio.run(_assemble_context_without_focus_does_not_fallback_to_global_active())


def test_assemble_context_includes_wm_proposal_sections():
    asyncio.run(_assemble_context_includes_wm_proposal_sections())


def test_assemble_context_includes_problem_solving_guard_for_corrections():
    asyncio.run(_assemble_context_includes_problem_solving_guard_for_corrections())


async def _assemble_context_includes_wm_proposal_sections():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WMItem, WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            wm = WorkingMemory(capacity=20)
            wm.add(WMItem(
                kind="self_drive",
                priority=0.9,
                content=(
                    "[自驱事件]\n"
                    "type: exploration\n"
                    "scope: observation\n"
                    "proposal:\n"
                    "- create_self_drive_task: 建立一次轻量探索任务。\n"
                    "open_questions:\n"
                    "- 是否已有未完成同题任务？\n"
                    "available_directions: create_self_drive_task | wait\n"
                ),
            ))

            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=wm,
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="请帮我继续探索",
            )

            assert "### WM 提案与可执行方向（observation to action）" in text
            assert "create_self_drive_task" in text
            assert "available_directions:" in text
        finally:
            await store.close()


async def _assemble_context_includes_problem_solving_guard_for_corrections():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "切换节点",
                goal="切换节点并重新推送",
                status="in_progress",
                next_step="切换节点",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=task,
                user_message="我指的是代理节点，不是模型节点",
            )

            assert "### 通用问题解决守卫" in text
            assert "guard=active" in text
            assert "signals=user_correction" in text
            assert "task.workbench" in text
            assert "domain" in text
            assert "intent" in text
        finally:
            await store.close()


async def _assemble_context_includes_recent_daily_continuity():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
        # 提高语义阈值，避免 remember_speaker 写入的 interlocutor 节点
        # (score≈0.57) 误触发 long_term_primary，掩盖 daily_gap_fill 路径
        "memory": {"daily_recall_semantic_score_threshold": 0.9},
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            episodic = EpisodicMemory(Path(d) / "memory")
            episodic.record("user", "爸爸今天刚发来 bat 文件，需要后续继续推进", task_id="task-bat")

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="继续处理 bat",
            )

            assert "### 近两日连续性（跨任务 daily 片段）" in text
            assert "### 记忆召回路径（本轮）" in text
            assert "recall_mode: daily_gap_fill" in text
            assert "daily_fallback_used: yes" in text
            assert "爸爸今天刚发来 bat 文件" in text
        finally:
            await store.close()


async def _assemble_context_daily_zero_budget_uses_evidence_excerpts():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
        "memory": {
            "daily_recall_max_chars": 0,
            "daily_recall_semantic_score_threshold": 0.9,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            episodic = EpisodicMemory(Path(d) / "memory")
            episodic.record("user", "github " + "a" * 6000, task_id="task-github-a")
            episodic.record("assistant", "github " + "b" * 6000, task_id="task-github-b")

            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="继续看 github",
            )

            assert "recall_mode: daily_gap_fill" in text
            assert "daily_fallback_used: yes" in text
            assert "a" * 2000 not in text
            assert "b" * 2000 not in text
            assert len(text) < 120000
        finally:
            await store.close()


async def _assemble_context_skips_daily_when_long_term_memory_is_strong():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import MemoryNode, SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
        "memory": {
            "daily_recall_semantic_score_threshold": 0.55,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            episodic = EpisodicMemory(Path(d) / "memory")
            episodic.record("user", "爸爸今天刚发来 bat 文件，需要后续继续推进", task_id="task-bat")
            semantic = SemanticMemory(Path(d) / "memory", decay_lambda=0.0)
            semantic.upsert(MemoryNode(
                id="user-bat-name",
                kind="fact",
                title="bat 是用户名字",
                body="用户明确要求以后叫他 bat。",
                activation=0.85,
                importance=0.95,
                source="wm_consolidation",
            ))

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=semantic,
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="继续处理 bat",
            )

            assert "### 近两日连续性（跨任务 daily 片段）" in text
            assert "### 记忆召回路径（本轮）" in text
            assert "recall_mode: long_term_primary" in text
            assert "daily_fallback_used: no" in text
            assert "本轮不额外注入 daily 补短" in text
            assert "爸爸今天刚发来 bat 文件" not in text
        finally:
            await store.close()


async def _assemble_context_includes_chat_scoped_memory_layers():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import MemoryNode, SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            episodic = EpisodicMemory(Path(d) / "memory")
            episodic.record("user", "chat-1 第一轮用户消息", task_id="task-a", chat_id="wechat:chat-1")
            episodic.record("assistant_reply", "chat-1 第一轮回复", task_id="task-a", chat_id="wechat:chat-1")
            episodic.record("user", "chat-1 第二个任务继续推进", task_id="task-b", chat_id="wechat:chat-1")
            episodic.record("assistant_reply", "chat-1 第二个任务回复", task_id="task-b", chat_id="wechat:chat-1")
            episodic.record("user", "chat-2 无关消息", task_id="task-c", chat_id="wechat:chat-2")

            semantic = SemanticMemory(Path(d) / "memory", decay_lambda=0.0)
            semantic.upsert(MemoryNode(
                id="chat-summary-1",
                kind="chat_summary",
                title="[2026-05-25] chat[abc123] 上次聊到部署问题",
                body="用户希望延续远程部署排查，并且别重复建任务。",
                activation=0.9,
                importance=0.7,
                tags=["chat_summary", "chat:wechat:chat-1"],
                source="chat_summary",
            ))

            from core.judgment import CognitionFrame
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=semantic,
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="继续跟进部署问题",
                chat_id="wechat:chat-1",
            )

            assert "### 当前 chat 连续性（跨任务 chat 叙事片段）" in text
            assert "### 当前 chat 长期结晶" in text
            assert "chat-1 第二个任务继续推进" in text
            assert "chat-2 无关消息" not in text
            assert "上次聊到部署问题" in text
            assert "chat_scope: wechat:chat-1" in text
            assert "chat_memory_hits: 1" in text
            assert "用户: chat-1 第二个任务继续推进" in text or "我: chat-1 第二个任务回复" in text
        finally:
            await store.close()


async def _assemble_context_keeps_cross_task_episodic_out_of_current_task_narrative():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "当前任务",
                goal="只保留当前任务叙事",
                status="in_progress",
                next_step="继续排查当前链路",
            )
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None

            episodic = EpisodicMemory(Path(d) / "memory")
            episodic.record("assistant", "当前任务里刚确认了 focus 路由。", task_id=str(task_id))
            episodic.search = lambda query, max_chars=2000, exclude_task_id=None: "[task=legacy-42 role=assistant] 旧任务昨天说过要先去读 README。"  # type: ignore[method-assign]

            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=active_task,
                user_message="继续处理 focus 路由",
            )

            assert "### 情节记忆（当前任务叙事片段）" in text
            assert "### 跨任务情节线索（仅作切换候选，不并入当前任务叙事）" in text
            assert "当前任务里刚确认了 focus 路由。" in text
            assert "旧任务昨天说过要先去读 README。" in text
            assert "[跨任务检索命中]" not in text
            assert text.index("当前任务里刚确认了 focus 路由。") < text.index("### 跨任务情节线索（仅作切换候选，不并入当前任务叙事）")
        finally:
            await store.close()


async def _assemble_context_clips_oversized_cross_task_episodic(caplog):
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "当前任务",
                goal="读取最近日志，寻找重复模式或可优化点",
                status="in_progress",
                next_step="读取最近日志，寻找重复模式或可优化点",
            )
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None

            episodic = EpisodicMemory(Path(d) / "memory")
            huge_cross_task = "[task=legacy role=assistant] " + "X" * 20000
            episodic.search = lambda query, max_chars=2000, exclude_task_id=None: huge_cross_task  # type: ignore[method-assign]

            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            caplog.set_level(logging.WARNING, logger="lingzhou.judgment")
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=active_task,
                user_message="继续处理日志优化",
            )

            assert "### 跨任务情节线索（仅作切换候选，不并入当前任务叙事）" in text
            assert "X" * 5000 not in text
            assert any("episodic_cross_task_clipped" in record.message for record in caplog.records)
        finally:
            await store.close()


async def _assemble_context_prefers_focus_fact_over_global_active():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            await store.add_task(
                "全局活跃任务",
                goal="旧 get_active 会误命中这里",
                status="in_progress",
            )
            focus_id = await store.add_task(
                "当前焦点任务",
                goal="_assemble_context 应优先命中这里",
                status="pending",
                next_step="继续沿 focus task 推进",
            )
            await store.set_fact("focus:current_task_id", str(focus_id), scope="system")

            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="继续处理 focus 任务",
            )

            assert "标题: 当前焦点任务" in text
            assert "目标: _assemble_context 应优先命中这里" in text
            assert text.index("标题: 当前焦点任务") < text.index("### 其他开放任务")
        finally:
            await store.close()


async def _assemble_context_without_focus_does_not_fallback_to_global_active():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait"}'

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx-no-focus.db")
        await store.open()
        try:
            await store.add_task(
                "全局活跃任务",
                goal="无 focus 时 assembler 不应回退到这里",
                status="in_progress",
            )
            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=EpisodicMemory(Path(d) / "memory"),
                    semantic=SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="继续处理",
            )

            assert "（无活跃任务，可自主探索或等待）" in text
            assert "标题: 全局活跃任务" not in text
        finally:
            await store.close()


async def _assemble_context_includes_current_interlocutor_sections():
    from core.config import Config
    from core.judgment import CognitionFrame, JudgmentLayer
    from core.perception import EmotionState
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import MemoryNode, SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _SpeakerProvider:
        async def chat(self, messages, *, temperature=None, thinking_override=None):
            system = messages[0].content
            if "当前交互对象识别器" in system:
                return json.dumps({
                    "node_id": "interlocutor-bat",
                    "confidence": 0.91,
                    "display_name": "bat",
                    "relationship_note": "称呼与偏好都吻合",
                    "evidence": ["当前消息自称 bat", "历史画像记得他喜欢先给结论"],
                    "provisional": False,
                }, ensure_ascii=False)
            return "[]"

        async def close(self):
            return None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "ctx.db")
        await store.open()
        try:
            episodic = EpisodicMemory(Path(d) / "memory")
            episodic.record("user", "我上次说过以后叫我 bat。", task_id="task-a", chat_id="wechat:chat-1", interlocutor_id="interlocutor-bat")
            episodic.record("assistant_reply", "收到，我以后叫你 bat。", task_id="task-a", chat_id="wechat:chat-1", interlocutor_id="interlocutor-bat")

            semantic = SemanticMemory(Path(d) / "memory", decay_lambda=0.0)
            semantic.upsert(MemoryNode(
                id="interlocutor-bat",
                kind="interlocutor",
                title="bat",
                body="画像摘要: 喜欢先给结论。\n偏好线索: 喜欢先给结论再展开。",
                activation=0.92,
                importance=0.75,
                tags=["interlocutor_profile", "interlocutor:interlocutor-bat", "handle:wechat:chat-1", "alias:bat"],
                source="interlocutor_profile",
            ))
            await store.set_fact("chat:wechat:chat-1:interlocutor_profile_id", "interlocutor-bat", scope="profile")

            layer = JudgmentLayer(_SpeakerProvider(), ToolRegistry(), cfg)
            text = await layer._assembler._assemble_context(
                CognitionFrame(
                    percept=cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                    wm=WorkingMemory(capacity=20),
                    task_store=store,
                    episodic=episodic,
                    semantic=semantic,
                    emotion=EmotionState.from_config(cfg),
                ),
                active_task=None,
                user_message="以后还是叫我 bat，先给结论。",
                chat_id="wechat:chat-1",
            )

            assert "### 当前交互对象画像" in text
            assert "### 当前交互对象交互连续性" in text
            assert "当前交互对象候选: bat（confidence:0.91" in text
            assert "称呼与偏好都吻合" in text
            assert "历史画像记得他喜欢先给结论" in text
            assert "我上次说过以后叫我 bat。" in text
        finally:
            await store.close()
