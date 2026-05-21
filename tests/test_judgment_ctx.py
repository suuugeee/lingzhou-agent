"""行为门控、技能、thinking、模型路由、任务改写等 judgment context 测试"""
import asyncio
import builtins
import io
import json
import logging
import math
import os
import tempfile
import time
from functools import lru_cache
from datetime import datetime, UTC, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiosqlite
import pytest

from conftest import (
    _proj_root,
    _test_config,
    _tool_ctx,
    _execution_layer,
    _tool_registry,
    _judgment_output,
)
# ══════════════════════════════════════════════════════════════════════════════
# SemanticMemory — 多锚点情境召回（ACT-R 收敛激活）
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_multi_anchor_convergence_bonus():
    """多锚点命中同一节点时 convergence_bonus 使其排名高于单锚点命中节点。

    设计原理：两节点在主锚点 "importlib" 上得分相近，但 node_ab 的 body
    同时命中第二锚点 "热加载 reload"，因此多锚点命中使其 final_score 更高。
    """
    from memory.semantic import SemanticMemory, MemoryNode

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
    from memory.semantic import SemanticMemory, MemoryNode

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.0)
        sm.upsert(MemoryNode(id="x", kind="fact", title="test", body="body", activation=0.5))

        assert sm.retrieve_multi_anchor([]) == []
        assert sm.retrieve_multi_anchor(["", "  "]) == []


def test_fill_template_raises_when_variable_missing():
    from core.judgment.context import _fill_template

    with pytest.raises(ValueError, match="missing_field"):
        _fill_template("hello {{ missing_field }}", {"other": "value"})


# ══════════════════════════════════════════════════════════════════════════════
# 今日新增功能验证
# ══════════════════════════════════════════════════════════════════════════════

def test_model_health_circuit_breaker_blocks_and_clears():
    """ModelHealth 断路器：标记冷却后 _is_model_available 返回 False，
    recover 后返回 True；fallback tier 在主 tier 冷却时被选中。"""
    import time
    from core.config import Config
    from core.judgment import JudgmentLayer, ModelHealth
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
    assert layer._is_model_available("bailian/qwen3.6-plus") is True

    # 标记 429 错误 → 进入冷却
    layer._mark_model_failure("bailian/qwen3.6-plus", "Client error '429 Too Many Requests'")
    assert layer._is_model_available("bailian/qwen3.6-plus") is False

    # recover → 可用
    health = layer._get_health("bailian/qwen3.6-plus")
    health.cooldown_until = time.time() - 1  # 手动过期
    assert layer._is_model_available("bailian/qwen3.6-plus") is True


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
    assert layer._select_tier(phase="initial", user_message="hello") == "reasoner"

    # repair phase → repair
    assert layer._select_tier(phase="repair", user_message="") == "repair"

    # prefer_tier 优先
    assert layer._select_tier(phase="initial", user_message="", prefer_tier="reader") == "reader"

    # continue + reader tool + no error → reader
    tier = layer._select_tier(
        phase="continue", user_message="",
        current_action="file.read", tool_history=[],
    )
    assert tier == "reader"

    # continue + reasoner tool → reasoner
    tier2 = layer._select_tier(
        phase="continue", user_message="",
        current_action="shell.run", tool_history=[],
    )
    assert tier2 == "reasoner"


def test_prefer_tier_for_task_uses_pending_then_task_default():
    from core.loop.common import _next_initial_tier_hint, _prefer_tier_for_task
    from memory.task_store import Task

    task = Task(
        id=1,
        title="任务",
        status="pending",
        priority="normal",
        created_at="2026-05-15T00:00:00Z",
        model_tier="reader",
    )

    assert _prefer_tier_for_task(None, task) is None
    assert _prefer_tier_for_task("reader", task) == "reader"
    assert _prefer_tier_for_task("repair", task) == "repair"

    task.model_tier = "reasoner"
    assert _prefer_tier_for_task(None, task) == "reasoner"

    task.model_tier = "invalid"
    assert _prefer_tier_for_task(None, task) is None

    assert _next_initial_tier_hint(_judgment_output(decision="act", chosen_action_id="file.read")) is None
    assert _next_initial_tier_hint(
        _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            model_strategy={"next_phase_tier": "reader"},
        )
    ) == "reader"


def test_behavior_gate_passthrough_and_logs_observation(caplog):
    """重复信号只做感知和日志，不替 LLM 改 decision。"""
    from core.behavior_tracker import BehaviorTracker
    from core.judgment import JudgmentOutput

    caplog.set_level(logging.INFO, logger="lingzhou.behavior_tracker")
    tracker = BehaviorTracker()

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
    assert gated is action
    assert any("delegated to llm" in rec.message for rec in caplog.records)

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
    assert gated_read is read_action

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
    )

    assert skills
    assert any("[skill.match] selected=" in rec.message for rec in caplog.records)


def test_judgment_skills_for_log_formats_selected_names():
    from core.judgment import JudgmentLayer
    from core.skill import Skill

    assert JudgmentLayer._skills_for_log([]) == "none"
    assert JudgmentLayer._skills_for_log([
        Skill(name="runtime.bootstrap", description="", guidance=""),
        Skill(name="task.continuity", description="", guidance=""),
    ]) == "runtime.bootstrap,task.continuity"


def test_behavior_list_result_aware():
    """file.list 应按“结果是否相同”而不是仅按路径判定重复。"""
    from core.behavior_tracker import BehaviorTracker

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
    from core.behavior_tracker import BehaviorTracker

    tracker = BehaviorTracker()
    items = []
    for _ in range(10):
        items = tracker.on_act("file.list", "/root", task_id=None)

    assert items == []


def test_next_thinking_override_is_one_shot_and_strict():
    from core.loop.common import _next_thinking_override

    assert _next_thinking_override({"thinking_override": "low"}) == "low"
    assert _next_thinking_override({"thinking_override": "invalid"}) is None
    assert _next_thinking_override({}) is None
    assert _next_thinking_override(None) is None


def test_resolve_thinking_override_uses_mode_defaults_and_strategy():
    from core.loop.common import _resolve_thinking_override

    cfg = cast(Any, SimpleNamespace(
        thinking="off",
        loop=SimpleNamespace(chat_thinking="low", autonomous_thinking="medium"),
    ))

    assert _resolve_thinking_override(cfg, user_message="hi") == "low"
    assert _resolve_thinking_override(cfg, user_message="") == "medium"
    assert _resolve_thinking_override(cfg, user_message="", pending_override="high") == "high"
    assert _resolve_thinking_override(cfg, user_message="", model_strategy={"thinking_override": "minimal"}) == "minimal"


def test_thinking_floor_respects_chat_minimum_for_user_message():
    from core.loop.common import _thinking_floor

    assert _thinking_floor("off", "low") == "low"
    assert _thinking_floor("minimal", "low") == "low"
    assert _thinking_floor("high", "low") == "high"
    assert _thinking_floor(None, "low") == "low"


def test_recent_runs_summary_prefers_output_and_progress():
    from core.judgment.context import _fmt_recent_runs
    from memory.task_store import Run

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


def test_waiting_tasks_section_exposes_wait_reason_and_next_step():
    from core.judgment.context import _fmt_waiting_tasks
    from memory.task_store import Task

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
    payload = json.loads(layer._build_model_routing_section(
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
    assert payload["implicit_next_phase_default"] is None


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
    await layer._ref_resolver._llm_reason(
        "继续上次的话题",
        {"n1": {"kind": "task", "title": "旧任务", "body": "body"}},
    )
    payload = json.loads(layer._build_model_routing_section(
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


def test_model_routing_section_exposes_implicit_reader_default():
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
    payload = json.loads(layer._build_model_routing_section(
        phase="continue",
        user_message="",
        current_action="file.read",
        tool_history=[{"tool": "file.read", "params": {"path": "/tmp/a"}, "result": "ok"}],
        effective_thinking="low",
    ))

    assert payload["implicit_next_phase_default"]["tier"] == "reader"
    assert payload["implicit_next_phase_default"]["trigger"] == "last_action=file.read"


def test_fmt_durable_failures_exposes_policy_and_muted_actions():
    from core.judgment.context import _fmt_durable_failures

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
    from core.judgment.context import _load_durable_failure_snapshot
    from memory.task_store import TaskStore

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
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "file.list", "params": {"path": "/tmp"}, "result": "ok"}],
        user_message="继续",
        prefer_tier="reasoner",
        thinking_override="low",
    )

    assert out.decision == "wait"
    assert provider.last_thinking_override == "low"
    assert layer.last_call_meta["thinking"] == "low"


def test_action_made_progress_result_aware():
    from core.judgment import JudgmentOutput
    from core.loop.progress import _action_made_progress, _result_fingerprint
    from tools.registry import ToolResult

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


def test_write_success_stall_meta_reflection_records_task_hint():
    asyncio.run(_write_success_stall_meta_reflection_records_task_hint())


async def _write_success_stall_meta_reflection_records_task_hint():
    from core.loop.postprocess import _write_success_stall_meta_reflection
    from memory.task_store import TaskStore
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


def test_fallback_reply_for_user_describes_waiting_state():
    from core.loop.logging import _fallback_reply_for_user
    from tools.registry import ToolResult
    from memory.task_store import Task

    action = _judgment_output(decision="act", chosen_action_id="task.wait", next_step="等用户补充路径后重新验证目录")
    result = ToolResult(
        summary="任务 [27] 已进入 waiting: external/source-path",
        state_delta={"task_status": "waiting", "wait_kind": "external", "wait_key": "source-path"},
    )
    task = Task(id=27, title="等待路径", status="in_progress", priority="normal", created_at="2026-05-15T14:00:00+00:00")

    reply = _fallback_reply_for_user(action, result, task)
    assert reply.startswith("状态: waiting")
    assert "waiting" in reply
    assert "external/source-path" in reply
    assert "等用户补充路径后重新验证目录" in reply


def test_fallback_reply_for_user_uses_real_error_instead_of_background_ack():
    from core.loop.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="pause", rationale="源路径证据不存在，需要用户补充。")
    result = ToolResult(summary="路径不存在: /root/.legacy-runtime/source", error="FileNotFound")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("状态: error")
    assert "detail:" in reply
    assert "路径不存在" in reply
    assert "后台继续处理" not in reply
    assert "我这轮" not in reply


def test_fallback_reply_for_user_does_not_echo_tool_summary_on_success():
    from core.loop.logging import _fallback_reply_for_user
    from tools.registry import ToolResult

    action = _judgment_output(decision="act", chosen_action_id="file.read", rationale="我已经收集到关键证据。")
    result = ToolResult(summary="/tmp/a.py\n/tmp/b.py")

    reply = _fallback_reply_for_user(action, result, None)
    assert reply.startswith("状态: progressed")
    assert "basis:" in reply
    assert "/tmp/a.py" not in reply


def test_infer_valence_from_text_uses_explicit_hint_only():
    from core.loop.common import _infer_valence_from_text

    assert _infer_valence_from_text("继续推进，暂无结构化情绪提示", 0.6) == 0.6
    assert _infer_valence_from_text("root cause found; valence=0.2", 0.6) == pytest.approx(0.52)


def test_should_continue_within_tick_for_autonomous_act():
    from core.judgment import JudgmentOutput
    from core.loop.common import _preferred_continue_tier, _should_continue_within_tick

    assert _should_continue_within_tick(_judgment_output(decision="act", chosen_action_id="file.read")) is True
    assert _should_continue_within_tick(_judgment_output(decision="act", chosen_action_id="task.complete")) is False
    assert _should_continue_within_tick(_judgment_output(decision="wait")) is False
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=True,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.read"),
        user_message="帮我看下 mini 为什么 400",
        has_active_task=False,
    ) is True
    assert _should_continue_within_tick(
        _judgment_output(decision="act", chosen_action_id="file.write"),
        user_message="帮我顺手改一下配置",
        has_active_task=True,
    ) is False
    assert _preferred_continue_tier(
        _judgment_output(decision="act", chosen_action_id="memory.search"),
        user_message="继续分析这个问题",
    ) == "reader"
    assert _preferred_continue_tier(
        _judgment_output(
            decision="act",
            chosen_action_id="memory.search",
            model_strategy={"next_phase_tier": "reader"},
        ),
        user_message="继续分析这个问题",
    ) == "reader"


def test_rewrite_task_ask_to_task_list_before_asking_for_id():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "请提供相关 task id"},
        rationale="我需要先定位任务。",
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="帮我看看昨晚那个任务为什么没回消息",
        registry=_tool_registry(),
    )

    assert rewritten.chosen_action_id == "task.list"
    assert rewritten.params == {"status": "all", "limit": 10}
    assert rewritten.reply_to_user == ""
    assert rewritten.model_strategy["next_phase_tier"] == "reasoner"


async def test_decide_continue_reply_only_forces_reasoner_and_reply_to_user():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

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
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 2 条相关记忆"}],
        user_message="继续分析",
        prefer_tier="reader",
        reply_only=True,
    )

    assert out.decision == "pause"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.reply_to_user == "这是最终回复。"
    assert layer.last_call_meta["tier"] == "reasoner"
    assert provider.last_messages is not None
    assert "禁止再调用任何工具" in provider.last_messages[1].content


async def test_decide_continue_includes_structured_tool_history_window():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

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
    layer._last_context_text = "cached context"

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


async def test_decide_continue_rewrites_complex_act_to_task_plan_without_existing_plan():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from memory.task_store import Task
    from tools.registry import ToolRegistry

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
    layer._last_context_text = "cached context"

    out = await layer.decide_continue(
        [{"tool": "memory.search", "params": {"query": "chat 回复"}, "result": "命中 2 条相关记忆"}],
        user_message="请你逐一排查并修复 chat 回复问题",
        active_task=task,
        prefer_tier="reasoner",
    )

    assert out.decision == "act"
    assert out.chosen_action_id == "task.plan"
    assert out.params["task_id"] == 11
    assert out.params["plan"][0]["status"] == "in_progress"
    assert out.params["plan"][1] == {"step": "再修复 chat 回复链路", "status": "pending"}


def test_rewrite_task_ask_keeps_direct_question_after_evidence():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "还需要你补充具体 id 吗？"},
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="继续分析",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 2 条相关记忆"},
            {"tool": "task.list", "params": {"status": "all"}, "result": "[12] [in_progress] 继续分析"},
        ],
        registry=_tool_registry(),
    )

    assert rewritten.chosen_action_id == "task.ask"
    assert rewritten.params["question"] == "还需要你补充具体 id 吗？"


def test_rewrite_task_ask_requires_evidence_budget_before_asking():
    from core.judgment.runtime import _rewrite_task_ask_to_evidence

    action = _judgment_output(
        decision="act",
        chosen_action_id="task.ask",
        params={"question": "还需要你补充具体 id 吗？"},
    )

    rewritten = _rewrite_task_ask_to_evidence(
        action,
        user_message="继续分析",
        tool_history=[
            {"tool": "memory.search", "params": {"query": "继续分析"}, "result": "命中 1 条相关记忆"},
        ],
        registry=_tool_registry(),
    )

    assert rewritten.chosen_action_id == "task.list"
    assert rewritten.params == {"status": "all", "limit": 10}
    assert "1/2" in rewritten.rationale


def test_rewrite_complex_user_act_to_task_plan_before_mutation():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task

    task = Task(
        id=7,
        title="修复 chat 回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="逐一排查 chat 回复链路",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="shell.run",
        params={"command": "rg \"chat\" core -n"},
        rationale="先检查日志，再修复回复链路。",
        next_step="再修复 chat 回复链路",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="请你逐一排查昨天日志并修复 chat 回复问题",
        active_task=task,
        registry=_tool_registry(),
    )

    assert rewritten.chosen_action_id == "task.plan"
    assert rewritten.params["task_id"] == 7
    assert rewritten.params["plan"][0]["status"] == "in_progress"
    assert rewritten.params["plan"][0]["step"].startswith("执行 shell.run")
    assert rewritten.params["plan"][1] == {"step": "再修复 chat 回复链路", "status": "pending"}
    assert rewritten.reply_to_user == ""


def test_rewrite_complex_user_act_keeps_read_action_without_plan():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task

    task = Task(
        id=8,
        title="排查日志",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="逐一分析昨天日志",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="file.read",
        params={"path": "/tmp/runtime.log"},
        next_step="再总结问题点",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="请你逐一分析昨天日志并整理问题",
        active_task=task,
        registry=_tool_registry(),
    )

    assert rewritten.chosen_action_id == "file.read"


def test_rewrite_complex_user_act_uses_structural_next_step_not_keyword_regex():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task

    task = Task(
        id=9,
        title="处理回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="处理 chat 回复",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="shell.run",
        params={"command": "pytest -q"},
        next_step="汇总结果并修复失败项",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="帮我处理一下",
        active_task=task,
        registry=_tool_registry(),
    )

    assert rewritten.chosen_action_id == "task.plan"
    assert rewritten.params["plan"][1] == {"step": "汇总结果并修复失败项", "status": "pending"}


def test_rewrite_complex_user_act_respects_plan_alignment_exempt_capability():
    from core.judgment.runtime import _rewrite_complex_act_to_task_plan
    from memory.task_store import Task
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="debug.plan.exempt",
        description="调试用对齐豁免工具",
        capabilities=("plan_alignment_exempt",),
    ))
    async def _debug_plan_exempt(params: dict[str, object], ctx: ToolContext) -> ToolResult:
        return ToolResult(summary="ok")

    task = Task(
        id=10,
        title="处理回复",
        status="active",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="处理 chat 回复",
    )
    action = _judgment_output(
        decision="act",
        chosen_action_id="debug.plan.exempt",
        params={"query": "check"},
        next_step="继续汇总",
    )

    rewritten = _rewrite_complex_act_to_task_plan(
        action,
        user_message="帮我继续处理",
        active_task=task,
        registry=ToolRegistry(),
    )

    assert rewritten.chosen_action_id == "debug.plan.exempt"


def test_preferred_continue_tier_uses_manifest_reader_tier():
    from core.loop.common import _preferred_continue_tier, _should_continue_within_tick
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

    assert _preferred_continue_tier(action, user_message="继续分析", registry=reg) == "reader"
    assert _should_continue_within_tick(
        action,
        user_message="继续分析",
        has_active_task=True,
        registry=reg,
    ) is True


async def test_sync_task_progress_state_promotes_previous_next_step():
    from core.task_runtime import _sync_task_progress_state
    from memory.task_store import TaskStore

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
    from core.task_runtime import _sync_task_progress_state
    from memory.task_store import TaskStore

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


def test_fmt_task_exposes_runtime_state_to_llm():
    from core.judgment.context import _fmt_task
    from memory.task_store import Task

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


async def _fmt_context_facts_surfaces_task_and_recent_general_facts():
    from core.judgment.context import _fmt_context_facts, _load_context_facts_snapshot
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / 'facts.db')
        await store.open()
        task_id = await store.add_task('分析旧运行时记忆', goal='确认 carrier')
        task = await store.get_task_by_id(task_id)
        assert task is not None

        await store.set_fact(f'task:{task_id}:progress', '已确认 sqlite 为主载体', scope='task')
        await store.set_fact('legacy_runtime.workspace_memory.primary_carrier', '/root/.legacy-runtime/memory/main.sqlite')
        await store.set_fact('pref:routing_overrides', '{"reader":"demo"}', scope='system')

        facts = await _load_context_facts_snapshot(store, task)
        text = _fmt_context_facts(facts)

        assert f'task:{task_id}:progress' in text
        assert 'legacy_runtime.workspace_memory.primary_carrier' in text
        assert 'pref:routing_overrides' not in text
        await store.close()


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


def test_clip_reply_for_log_strips_memory_context():
    from core.loop.logging import _clip_reply_for_log

    clipped = _clip_reply_for_log("<memory-context>hidden</memory-context>\n用户可见回复")
    assert clipped == "用户可见回复"


def test_assemble_context_prefers_active_task_override_with_inbox():
    asyncio.run(_assemble_context_prefers_active_task_override_with_inbox())


async def _assemble_context_prefers_active_task_override_with_inbox():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from core.perception import EmotionState
    from memory.episodic import EpisodicMemory
    from memory.semantic import SemanticMemory
    from memory.task_store import TaskStore
    from memory.working import WorkingMemory
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
            task.extras["inbox_messages"] = ["收到新的用户指令：请你使用 puppeteer 去搜索。"]

            layer = JudgmentLayer(_DummyProvider(), ToolRegistry(), cfg)
            text = await layer._assemble_context(
                cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False)),
                WorkingMemory(capacity=20),
                store,
                EpisodicMemory(Path(d) / "memory"),
                SemanticMemory(Path(d) / "memory", decay_lambda=0.0),
                EmotionState.from_config(cfg),
                active_task=task,
                user_message="请你使用 puppeteer 去搜索。",
            )

            assert "收到新的用户指令：请你使用 puppeteer 去搜索。" in text
        finally:
            await store.close()


