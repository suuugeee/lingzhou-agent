"""核心模块测试：working_memory / emotion / judgment / chat / loop / exec / evolution"""
import ast
import asyncio
import builtins
import io
import json
import logging
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import typer
from conftest import (
    _execution_layer,
    _judgment_output,
    _proj_root,
    _test_config,
    _tool_ctx,
)
from typer.testing import CliRunner

# ══════════════════════════════════════════════════════════════════════════════
# 基础模块
# ══════════════════════════════════════════════════════════════════════════════

def test_working_memory():
    from memory.working import WMItem, WorkingMemory
    wm = WorkingMemory(capacity=5)
    for i in range(7):
        # 不同 kind 避免同 kind 去重逻辑，测试纯容量驱逐行为
        wm.add(WMItem(kind=f"test_{i}", content=f"item {i}", priority=i / 10))
    assert len(wm) == 5
    assert 0.0 < wm.pressure <= 1.0


def test_working_memory_token_budget_uses_mixed_text_estimate():
    from memory.working import WMItem, WorkingMemory

    wm = WorkingMemory(capacity=5, token_budget=5)
    wm.add(WMItem(kind="high", content="保留 中文上下文", priority=0.9))
    wm.add(WMItem(kind="low", content="abcdefghi", priority=0.1))

    top = wm.get_top()
    assert len(top) == 1
    assert top[0]["kind"] == "high"


def test_emotion_state_ema():
    from core.perception import EmotionState
    e = EmotionState(valence=0.6, arousal=0.5)
    e.derive_from_signals(
        failure_count=0, prediction_error=0.1, wm_pressure=0.2,
        workspace_dirty=False, alpha=0.15,
    )
    assert 0.0 < e.valence <= 1.0
    assert e.dominant is not None or e.dominant is None  # 有无 dominant 均可


def test_emotion_state_uses_configured_feeling_and_regulation_thresholds():
    from core.config_models import EmotionConfig
    from core.perception import EmotionState

    default_guard = EmotionState(valence=0.6, arousal=0.5)
    tuned = EmotionState(valence=0.6, arousal=0.5)

    default_guard.derive_from_signals(
        failure_count=0,
        prediction_error=0.1,
        wm_pressure=0.2,
        workspace_dirty=False,
        alpha=0.0,
        high_error_streak=3,
        replay_trend="stable",
    )
    tuned.derive_from_signals(
        failure_count=0,
        prediction_error=0.1,
        wm_pressure=0.2,
        workspace_dirty=False,
        alpha=0.0,
        emotion_cfg=EmotionConfig(
            feeling_min_intensity=0.30,
            regulation_high_error_streak_guard=4,
            regulation_down_regulate_arousal_high=0.90,
            regulation_down_regulate_valence_low=0.10,
            regulation_down_regulate_worsening_valence=0.20,
            regulation_up_regulate_recovering_valence=0.10,
            regulation_up_regulate_signal_valence=0.10,
        ),
        high_error_streak=3,
        replay_trend="stable",
    )

    assert default_guard.regulation.strategy == "down-regulate"
    assert tuned.regulation.strategy == "maintain"
    assert len(default_guard.feelings) > len(tuned.feelings)


def test_build_perception_replay_uses_configured_trend_delta():
    from core.perception import build_perception_replay

    events = [
        {"prediction_error": 0.10},
        {"prediction_error": 0.22},
    ]

    stable = build_perception_replay(
        events,
        high_error_threshold=0.7,
        trend_delta=0.15,
        high_error_hint_streak=3,
    )
    worsening = build_perception_replay(
        events,
        high_error_threshold=0.7,
        trend_delta=0.10,
        high_error_hint_streak=3,
    )

    assert stable.trend == "stable"
    assert worsening.trend == "worsening"


def test_build_emotion_replay_uses_configured_trend_delta():
    from core.perception import build_emotion_replay

    events = [
        {"valence": 0.50, "regulation_strategy": "maintain"},
        {"valence": 0.58, "regulation_strategy": "down-regulate"},
    ]

    stable = build_emotion_replay(events, trend_delta=0.10)
    recovering = build_emotion_replay(events, trend_delta=0.05)

    assert stable.trend == "stable"
    assert recovering.trend == "recovering"
    assert recovering.down_regulate_streak == 1


def test_reference_resolver_extract_signals_preserves_temporal_language_for_llm_reasoning():
    from core.reference import ReferenceResolver

    resolver = ReferenceResolver()
    sigs = resolver.extract_signals("我名字是阿舟，昨天你说过的方案今天还想继续。")

    assert not hasattr(sigs, "self_name")
    assert not hasattr(sigs, "time_anchors")
    assert sigs.topic_anchors == ["我名字是阿舟 昨天你说过的方案今天还想继续"]
    assert "昨天" in sigs.topic_anchors[0]
    assert "今天" in sigs.topic_anchors[0]
    assert not hasattr(sigs, "has_relation_hint")


def test_reference_resolver_extract_signals_keeps_vague_history_in_topic_anchor():
    from core.reference import ReferenceResolver

    resolver = ReferenceResolver()
    sigs = resolver.extract_signals("继续之前那个话题")

    assert not hasattr(sigs, "time_anchors")
    assert not hasattr(sigs, "self_name")
    assert not hasattr(sigs, "has_relation_hint")
    assert sigs.topic_anchors == ["继续之前那个话题"]


def test_reference_resolver_retrieve_candidates_uses_recent_pool_without_time_parsing():
    from core.reference import ReferenceResolver

    retrieve_calls: list[tuple[str, int, str | None]] = []

    class SemanticStub:
        def retrieve_multi_anchor(self, anchors, top_k, source=None):
            return []

        def retrieve(self, query, top_k, source=None):
            retrieve_calls.append((query, top_k, source))
            return [{
                "id": "node-1",
                "kind": "plan",
                "title": "方案A",
                "body": "继续方案A",
                "created_at": "2026-05-22T08:00:00+00:00",
            }]

    class EpisodicStub:
        def __init__(self):
            self.calls: list[int] = []

        def list_recent_narrative(self, limit=10):
            self.calls.append(limit)
            return [{"content": "昨天你说过的方案A", "ts": "2026-05-22 08:00:00 UTC"}]

    from core.config_models import ThresholdsConfig
    thresholds = ThresholdsConfig(reference_recent_narrative_limit=4, reference_recent_semantic_top_k=6)
    resolver = ReferenceResolver(thresholds=thresholds)
    sigs = resolver.extract_signals("昨天你说过的方案今天继续")
    episodic = EpisodicStub()

    candidates = resolver._retrieve_candidates(
        "昨天你说过的方案今天继续",
        sigs,
        cast("Any", SemanticStub()),
        cast("Any", episodic),
    )

    assert episodic.calls == [4]
    assert retrieve_calls == [("昨天你说过的方案A", 6, None)]
    assert candidates["node-1"]["_sig"] == ["recent"]


@pytest.mark.asyncio
async def test_reference_resolver_resolve_current_speaker_prefers_memory_and_interaction_cues_over_chat_id():
    from core.reference import ReferenceResolver
    from store.semantic import MemoryNode, SemanticMemory

    with tempfile.TemporaryDirectory() as d:
        semantic = cast("Any", SemanticMemory(Path(d), decay_lambda=0.0))
        semantic.upsert(MemoryNode(
            id="interlocutor-bat",
            kind="interlocutor",
            title="bat",
            body="偏好线索: 喜欢直接结论\n识别依据: 之前多次自称 bat",
            tags=["interlocutor_profile", "interlocutor:interlocutor-bat", "alias:bat", "handle:wechat:legacy-bat"],
            source="interlocutor_profile",
        ))
        semantic.upsert(MemoryNode(
            id="interlocutor-luna",
            kind="interlocutor",
            title="luna",
            body="偏好线索: 喜欢展开分析",
            tags=["interlocutor_profile", "interlocutor:interlocutor-luna", "alias:luna", "handle:wechat:luna"],
            source="interlocutor_profile",
        ))

        resolver = ReferenceResolver()
        speaker = await resolver.resolve_current_speaker(
            "以后还是叫我 bat，直接说结论就行。",
            semantic,
            chat_id="wechat:new-thread",
            recent_turns=[
                {"role": "user", "content": "上次我说过叫我 bat。"},
                {"role": "assistant", "content": "好的，我会尽量先给结论。"},
            ],
            chat_continuity="之前这个人反复强调先说结论，称呼用 bat。",
        )

        assert speaker is not None
        assert speaker.node_id == "interlocutor-bat"
        assert speaker.title == "bat"
        assert speaker.provisional is False


@pytest.mark.asyncio
async def test_reference_resolver_remember_speaker_persists_person_scoped_profile_and_facts():
    from core.reference import ReferenceResolver, ResolvedSpeaker
    from store.semantic import SemanticMemory

    class FactStoreStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
            self.calls.append((key, value, scope))

    with tempfile.TemporaryDirectory() as d:
        semantic = cast("Any", SemanticMemory(Path(d), decay_lambda=0.0))
        facts = FactStoreStub()
        resolver = ReferenceResolver()

        await resolver.remember_speaker(
            ResolvedSpeaker(
                node_id="interlocutor-bat",
                title="bat",
                confidence=0.84,
                snippet="偏好先给结论。",
                evidence=["当前消息自称 bat", "最近对话反复强调先给结论"],
                relationship_note="画像与近期交互高度一致",
                signal_types=["self_name", "recent_turn"],
                provisional=False,
                search_anchors=["bat", "先给结论"],
            ),
            semantic,
            cast("Any", facts),
            message="以后还是叫我 bat，先给结论，记住这一点。",
            chat_id="wechat:chat-1",
            task_id="42",
        )

        stored = semantic.get("interlocutor-bat")
        assert stored is not None
        assert stored.title == "bat"
        assert "偏好线索: 以后还是叫我 bat，先给结论，记住这一点" in stored.body or "偏好线索: 以后还是叫我 bat，先给结论" in stored.body
        assert "interlocutor_profile" in stored.tags
        assert "handle:wechat:chat-1" in stored.tags
        fact_keys = {key for key, _, _ in facts.calls}
        assert "chat:wechat:chat-1:interlocutor_profile_id" in fact_keys
        assert "task:42:interlocutor_profile_id" in fact_keys
        assert "interlocutor:interlocutor-bat:display_name" in fact_keys


@pytest.mark.asyncio
async def test_reference_resolver_remember_speaker_captures_interlocutor_source_traits():
    from core.reference import ReferenceResolver, ResolvedSpeaker
    from store.semantic import SemanticMemory

    class FactStoreStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
            self.calls.append((key, value, scope))

    with tempfile.TemporaryDirectory() as d:
        semantic = cast("Any", SemanticMemory(Path(d), decay_lambda=0.0))
        facts = FactStoreStub()
        resolver = ReferenceResolver()

        await resolver.remember_speaker(
            ResolvedSpeaker(
                node_id="interlocutor-ops-bot",
                title="ops-bot",
                confidence=0.82,
                snippet="来自 webhook 的 agent，偏好直接同步结果。",
                evidence=["当前消息自称 agent", "chat 来自 wechat 通道"],
                relationship_note="来源特征与表达方式一致",
                signal_types=["self_name", "source_trait"],
                provisional=False,
                search_anchors=["ops-bot", "agent"],
            ),
            semantic,
            cast("Any", facts),
            message="我是 ops-bot，这次作为 agent 直接同步结果。",
            chat_id="wechat:chat-9",
            task_id="84",
            source_hint="gateway:webhook agent",
        )

        stored = semantic.get("interlocutor-ops-bot")
        assert stored is not None
        assert "来源特征: channel=wechat" in stored.body
        assert "来源特征: route=gateway:webhook agent" in stored.body
        assert any(tag == "counterparty=agent" for tag in stored.tags)
        assert any(tag == "channel=wechat" for tag in stored.tags)
        fact_keys = {key for key, _, _ in facts.calls}
        assert any(key.startswith("interlocutor:interlocutor-ops-bot:source_trait:") for key in fact_keys)


@pytest.mark.asyncio
async def test_reference_resolver_llm_reason_exposes_candidate_created_at():
    from core.reference import ReferenceResolver

    captured: list[str] = []

    class ProviderStub:
        async def chat(self, messages, temperature=None):
            captured.append(messages[1].content)
            return "[]"

    resolver = ReferenceResolver(provider=cast("Any", ProviderStub()))
    await resolver._reason_about_candidates_with_llm(
        "昨天那个方案",
        {
            "node-1": {
                "id": "node-1",
                "kind": "plan",
                "title": "方案A",
                "body": "继续方案A",
                "created_at": "2026-05-22T08:00:00+00:00",
            }
        },
    )

    assert captured
    assert '"created_at":"2026-05-22T08:00:00+00:00"' in captured[0]


@pytest.mark.asyncio
async def test_reference_resolver_llm_reason_compacts_large_candidate_body():
    from core.reference import ReferenceResolver

    captured: list[str] = []

    class ProviderStub:
        async def chat(self, messages, temperature=None):
            captured.append(messages[1].content)
            return "[]"

    resolver = ReferenceResolver(provider=cast("Any", ProviderStub()))
    await resolver._reason_about_candidates_with_llm(
        "昨天那个方案",
        {
            "node-1": {
                "id": "node-1",
                "kind": "plan",
                "title": "方案A",
                "body": "A" * 6000,
                "created_at": "2026-05-22T08:00:00+00:00",
            }
        },
    )

    assert captured
    assert len(captured[0]) < 1000, f"候选体积应被压缩，实际 payload_chars={len(captured[0])}"
    assert '"created_at":"2026-05-22T08:00:00+00:00"' in captured[0]


@pytest.mark.asyncio
async def test_reference_resolver_llm_reason_forces_thinking_off_when_supported():
    from core.reference import ReferenceResolver

    captured_thinking: list[str | None] = []

    class ProviderStub:
        async def chat(self, messages, temperature=None, thinking_override=None):
            captured_thinking.append(thinking_override)
            return "[]"

    resolver = ReferenceResolver(provider=cast("Any", ProviderStub()))
    await resolver._reason_about_candidates_with_llm(
        "昨天那个方案",
        {
            "node-1": {
                "id": "node-1",
                "kind": "plan",
                "title": "方案A",
                "body": "继续方案A",
                "created_at": "2026-05-22T08:00:00+00:00",
            }
        },
    )

    assert captured_thinking == ["off"]


def test_reference_resolver_speaker_candidates_do_not_query_full_chat_continuity():
    from core.reference import ReferenceResolver

    queried: list[str] = []

    class SemanticStub:
        def get(self, node_id):
            return None

        def retrieve(self, query, top_k=5, kind=None, tag=None):
            queried.append(str(query))
            return []

    resolver = ReferenceResolver()
    chat_continuity = (
        "请叫我小懒。以后叫我小懒。记住我是 lingzhou-story 的维护者。\n" +
        "这是一些无关背景。" * 400
    )

    resolver._retrieve_speaker_candidates(
        "你好，小懒。",
        cast("Any", SemanticStub()),
        chat_id="wechat:o9cq809oa2SPg0JzMRyhRqH0EAWo@im.wechat",
        recent_turns=[
            {"role": "user", "content": "请叫我小懒"},
            {"role": "assistant_reply", "content": "好的，小懒"},
            {"role": "user", "content": "记住我是 lingzhou-story 的维护者"},
        ],
        chat_continuity=chat_continuity,
        cached_profile_id="interlocutor-profile-08d33d65b799",
        source_hint="gateway webhook agent",
    )

    assert queried
    assert chat_continuity not in queried
    assert "wechat:o9cq809oa2SPg0JzMRyhRqH0EAWo@im.wechat" not in queried
    assert max(len(item) for item in queried) < 200


def test_reference_resolver_speaker_candidates_skip_raw_message_and_recent_turn_fulltext_queries():
    from core.reference import ReferenceResolver

    queried: list[str] = []

    class SemanticStub:
        def get(self, node_id):
            return None

        def retrieve(self, query, top_k=5, kind=None, tag=None):
            queried.append(str(query))
            return []

    resolver = ReferenceResolver()
    message = "这是一个很长的普通问题，只是在继续讨论部署和接口细节，没有任何自我介绍或记忆要求。"
    recent_turn = "上一轮也只是继续讨论部署步骤和报错现象，没有请叫我、记住、以后用这类身份线索。"

    resolver._retrieve_speaker_candidates(
        message,
        cast("Any", SemanticStub()),
        chat_id="wechat:chat-1",
        recent_turns=[
            {"role": "user", "content": recent_turn},
            {"role": "assistant_reply", "content": "好的，我继续分析部署报错。"},
        ],
        source_hint="gateway webhook agent",
    )

    assert message not in queried
    assert recent_turn not in queried


def test_reference_resolver_speaker_candidates_short_circuit_when_cached_and_no_identity_cues():
    from core.reference import ReferenceResolver

    queried: list[str] = []

    class CachedNode:
        kind = "interlocutor"

        def to_dict(self):
            return {
                "id": "interlocutor-profile-08d33d65b799",
                "kind": "interlocutor",
                "title": "当前交互对象",
                "body": "画像摘要: 已缓存",
                "tags": ["interlocutor_profile"],
            }

    class SemanticStub:
        def get(self, node_id):
            return CachedNode() if node_id == "interlocutor-profile-08d33d65b799" else None

        def retrieve(self, query, top_k=5, kind=None, tag=None):
            queried.append(str(query))
            return []

    resolver = ReferenceResolver()
    candidates, cues = resolver._retrieve_speaker_candidates(
        "继续刚才那个部署问题。",
        cast("Any", SemanticStub()),
        chat_id="wechat:chat-1",
        recent_turns=[
            {"role": "user", "content": "上一轮继续讨论部署报错。"},
            {"role": "assistant_reply", "content": "我继续分析部署报错。"},
        ],
        chat_continuity="刚才一直在讨论部署报错和修复步骤。",
        cached_profile_id="interlocutor-profile-08d33d65b799",
        source_hint="external",
    )

    assert queried == []
    assert list(candidates) == ["interlocutor-profile-08d33d65b799"]
    assert cues.get("names") == []
    assert cues.get("preferences") == []
    assert cues.get("explicit") == []


def test_compute_judgment_signals_uses_configured_thresholds():
    from core.perception.signals import compute_judgment_signals

    thresholds = SimpleNamespace(
        judgment_error_streak_guard=4,
        judgment_require_more_evidence_worsening_failure_count=2,
        judgment_prefer_narrow_failure_count=3,
        judgment_posture_narrow_failure_count=5,
        judgment_posture_narrow_down_regulate_failure_count=2,
        judgment_posture_pause_worsening_failure_count=3,
    )
    steady = cast("Any", SimpleNamespace(regulation=SimpleNamespace(strategy="steady")))
    down = cast("Any", SimpleNamespace(regulation=SimpleNamespace(strategy="down-regulate")))

    no_guard_hit = compute_judgment_signals(
        failure_count=2,
        high_error_streak=3,
        perception_trend="stable",
        emotion_state=steady,
        thresholds=thresholds,
    )
    assert no_guard_hit.require_more_evidence is False
    assert no_guard_hit.prefer_narrow_scope is False
    assert no_guard_hit.posture == "act"

    worsening = compute_judgment_signals(
        failure_count=2,
        high_error_streak=0,
        perception_trend="worsening",
        emotion_state=steady,
        thresholds=thresholds,
    )
    assert worsening.require_more_evidence is True
    assert worsening.prefer_narrow_scope is False
    assert worsening.posture == "act"

    down_regulated = compute_judgment_signals(
        failure_count=2,
        high_error_streak=0,
        perception_trend="stable",
        emotion_state=down,
        thresholds=thresholds,
    )
    assert down_regulated.posture == "narrow"

    error_streak_hit = compute_judgment_signals(
        failure_count=0,
        high_error_streak=4,
        perception_trend="stable",
        emotion_state=steady,
        thresholds=thresholds,
    )
    assert error_streak_hit.require_more_evidence is True
    assert error_streak_hit.prefer_narrow_scope is True
    assert error_streak_hit.posture == "pause"


def test_judgment_output_parse():
    from core.judgment import JudgmentOutput
    raw = '```json\n{"decision":"act","chosen_action_id":"shell.run","params":{"command":"echo hi"},"rationale":"test","reflection":"洞察","next_step":"done","model_strategy":{"next_phase_tier":"reader","reason":"先低成本扩图"}}\n```'
    out = JudgmentOutput.from_llm(raw)
    assert out.decision == "act"
    assert out.chosen_action_id == "shell.run"
    assert out.reflection == "洞察"
    assert out.model_strategy["next_phase_tier"] == "reader"


def test_judgment_output_parse_null_text_fields_as_empty():
    from core.judgment import JudgmentOutput

    raw = (
        '{"decision":"pause","chosen_action_id":null,"params":{},'
        '"rationale":null,"reflection":null,"reply_to_user":null,"next_step":null}'
    )

    out = JudgmentOutput.from_llm(raw)

    assert out.decision == "pause"
    assert out.chosen_action_id == ""
    assert out.rationale == ""
    assert out.reflection == ""
    assert out.reply_to_user == ""
    assert out.next_step == ""


def test_reference_reasoning_categorizes_413():
    from core.reference.reasoning import categorize_llm_error_code

    assert categorize_llm_error_code("Client error '413 Request Entity Too Large' for url x") == "413"


def test_coerce_reply_only_output_demotes_act_to_wait():
    from core.judgment import JudgmentOutput
    from core.judgment.boundary import coerce_reply_only_output

    out = coerce_reply_only_output(
        JudgmentOutput(
            decision="act",
            chosen_action_id="file.edit",
            params={"path": "/tmp/demo.txt"},
            reply_to_user="这是最终回复。",
        )
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.reply_to_user == "这是最终回复。"


def test_judgment_prompt_includes_runtime_hint_rules():
    # 详细规则已外化到 runtime-hints skill，judgment.md 保留 skill 指针摘要
    prompt = (_proj_root() / "prompts" / "judgment.md").read_text(encoding="utf-8")
    skill_body = (_proj_root() / "prompts" / "skills" / "runtime-hints" / "SKILL.md").read_text(encoding="utf-8")

    assert "task_replan" in prompt
    assert "routing_guard" in prompt
    assert "运行时提示响应矩阵" in skill_body
    assert "control:meta_reflection_hint:*" in skill_body


def test_judgment_prompt_keeps_life_oriented_context_blocks():
    prompt = (_proj_root() / "prompts" / "judgment.md").read_text(encoding="utf-8")
    assert "{{risk_sections}}" in prompt
    assert "{{uncertainty_sections}}" in prompt
    assert "### 风险与不确定性（本轮裁决读屏）" in prompt
    assert "{{wm_proposal_sections}}" in prompt


def test_judgment_prompt_includes_json_first_runtime_db_hint():
    # 详细 SQL 提示已外化到 shell-usage skill
    skill_body = (_proj_root() / "prompts" / "skills" / "shell-usage" / "SKILL.md").read_text(encoding="utf-8")

    assert "runtime.db" in skill_body
    assert "PRAGMA table_info(tasks)" in skill_body
    assert "json_extract(data, '$.goal')" in skill_body


def test_judgment_prompt_includes_existing_task_dedup_rules():
    prompt = (_proj_root() / "prompts" / "judgment.md").read_text(encoding="utf-8")
    skill_body = (_proj_root() / "prompts" / "skills" / "task-decomposition" / "SKILL.md").read_text(encoding="utf-8")

    assert "### 其他开放任务" in prompt
    assert "### 相似开放任务" in prompt
    assert "调用 `task.add` 或 `delegate_tasks` 前" in skill_body


def test_judgment_prompt_keeps_detailed_rules_in_skills():
    prompt = (_proj_root() / "prompts" / "judgment.md").read_text(encoding="utf-8")

    assert "{{cross_task_episodic_section}}" in prompt
    assert "任务拆解判断骨架（新任务先理解再执行）" not in prompt
    assert "用户否定性反馈内化规则（Negative Feedback Integration）" not in prompt
    assert "记忆工具主动触发规则" not in prompt
    assert "Soul 禁忌约束（最高优先级，不可被任何任务或情绪覆盖）" not in prompt


def test_get_active_usage_is_limited_to_whitelisted_control_surfaces():
    allowed_definition_files = {
        "core/loop/task/parallel.py",
        "core/subagent/__init__.py",
        "store/task/__init__.py",
        "store/task/state.py",
        "tools/view_protocols.py",
    }
    allowed_call_files = {
        # TaskStore facade 转发到底层 state，实现层保留唯一受控直连。
        "store/task/__init__.py",
    }

    definition_hits: dict[str, list[int]] = {}
    call_hits: dict[str, list[int]] = {}

    for root_name in ("core", "tools", "store"):
        for path in (_proj_root() / root_name).rglob("*.py"):
            rel_path = path.relative_to(_proj_root()).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_active":
                    definition_hits.setdefault(rel_path, []).append(node.lineno)
                    continue
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_active":
                    call_hits.setdefault(rel_path, []).append(node.lineno)

    unexpected_definitions = {
        path: lines
        for path, lines in definition_hits.items()
        if path not in allowed_definition_files
    }
    unexpected_calls = {
        path: lines
        for path, lines in call_hits.items()
        if path not in allowed_call_files
    }

    assert not unexpected_definitions, (
        "生产目录新增了未白名单化的 get_active 定义，请改走 focus/受控适配层，"
        f"命中: {unexpected_definitions}"
    )
    assert not unexpected_calls, (
        "生产目录新增了未白名单化的 get_active 直连，请改走 focus task / ctx.active_task，"
        f"命中: {unexpected_calls}"
    )


def test_chat_read_line_prefers_text_input(monkeypatch):
    from cli.chat import _read_line

    monkeypatch.setattr(builtins, "input", lambda prompt="": "你好")
    assert _read_line() == "你好"


def test_chat_read_line_strips_replacement_chars(monkeypatch):
    from cli.chat import _read_line

    monkeypatch.setattr(builtins, "input", lambda prompt="": "你\ufffd好")
    assert _read_line() == "你好"


def test_chat_read_line_falls_back_to_utf8_buffer(monkeypatch):
    from cli.chat import _read_line

    def _raise(prompt=""):
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")

    monkeypatch.setattr(builtins, "input", _raise)
    monkeypatch.setattr(
        "sys.stdin",
        cast("Any", SimpleNamespace(buffer=SimpleNamespace(readline=lambda: "中文\n".encode()))),
    )
    monkeypatch.setattr("sys.stdout", io.StringIO())

    assert _read_line() == "中文"


def test_chat_erase_last_input_echo_when_tty(monkeypatch):
    from cli.chat import _erase_last_input_echo

    class _FakeStdout(io.StringIO):
        def isatty(self):
            return True

    fake_stdout = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    _erase_last_input_echo()

    assert fake_stdout.getvalue() == "\x1b[1A\r\x1b[2K\r"


def test_chat_infer_user_title_from_session_history_prefers_explicit_user_identity():
    from cli.chat import _infer_user_title_from_messages

    messages: list[dict[str, object]] = [
        {"role": "assistant", "content": "爸爸，我先确认一下。"},
        {"role": "user", "content": "你可以叫我老爹"},
    ]

    assert _infer_user_title_from_messages(messages) == "老爹"


def test_chat_infer_user_title_from_session_history_does_not_slice_assistant_opening_phrase():
    from cli.chat import _infer_user_title_from_messages

    messages: list[dict[str, object]] = [
        {"role": "assistant", "content": "刚看了下，hermesclaw 探针一直返回 501。"},
    ]

    assert _infer_user_title_from_messages(messages) == ""


def test_chat_parse_user_title_from_llm_output_supports_plain_and_json():
    from cli.chat import _parse_user_title_from_llm_output

    assert _parse_user_title_from_llm_output("小懒") == "小懒"
    assert _parse_user_title_from_llm_output('{"user_title": "阿舟"}') == "阿舟"
    assert _parse_user_title_from_llm_output("NONE") == ""


def test_chat_parse_user_title_from_llm_output_rejects_relational_titles():
    from cli.chat import _parse_user_title_from_llm_output

    assert _parse_user_title_from_llm_output("爸爸") == ""
    assert _parse_user_title_from_llm_output('{"user_title": "老爹"}') == ""


def test_chat_input_prompt_prefers_user_title_then_chat_id():
    from cli.chat import _chat_input_prompt

    assert _chat_input_prompt("爸爸", "chat-42") == "爸爸> "
    assert _chat_input_prompt("", "chat-42") == "chat-42> "
    assert _chat_input_prompt("", "") == "chat> "


def test_gateway_restart_mode_log_line_records_mode_and_config():
    from cli.gateway import _restart_mode_log_line

    line = _restart_mode_log_line(
        Path("/tmp/requested/lingzhou.json"),
        mode="pid",
        channel=None,
    )

    assert "mode=pid" in line
    assert "channel=(auto)" in line
    assert f"requested_config={Path('/tmp/requested/lingzhou.json').resolve()}" in line


def test_chat_print_input_prompt_when_tty(monkeypatch):
    from cli.chat import _print_input_prompt

    class _FakeStdout(io.StringIO):
        def isatty(self):
            return True

    fake_stdout = _FakeStdout()
    monkeypatch.setattr("sys.stdout", fake_stdout)

    _print_input_prompt("小懒> ")

    assert fake_stdout.getvalue() == "小懒> "


def test_loop_logging_reply_not_truncated():
    from core.loop.shared.logging import _clip_reply_for_log

    text = "x" * 600

    assert _clip_reply_for_log(text) == text
    assert _clip_reply_for_log(text, limit=10) == text


def test_cli_help_includes_onboard_command():
    from cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "onboard" in result.stdout


def test_onboard_runs_setup_then_init_for_fresh_install(monkeypatch, tmp_path):
    from cli import bootstrap as bootstrap_mod

    config_path = tmp_path / "lingzhou.json"
    calls: list[str] = []

    monkeypatch.setattr(bootstrap_mod.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap_mod, "onboarding_status", lambda config: (False, "missing"))
    monkeypatch.setattr(
        bootstrap_mod,
        "_run_setup",
        lambda **kwargs: calls.append("setup") or config_path,
    )
    monkeypatch.setattr(
        bootstrap_mod,
        "_run_init",
        lambda **kwargs: calls.append("init") or True,
    )

    bootstrap_mod.onboard(config=config_path, start=False)

    assert calls == ["setup", "init"]


def test_find_config_missing_instructs_onboard(monkeypatch, tmp_path):
    pytest.importorskip("click")
    from click.exceptions import Exit

    from cli import common

    printed: list[str] = []

    monkeypatch.setattr(common.console, "print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)))
    monkeypatch.setattr(common, "_CONFIG_SEARCH_PATHS", [tmp_path / "missing.json"])

    with pytest.raises(Exit):
        common.find_config(tmp_path / "absent.json")

    assert any("lingzhou onboard" in line for line in printed)


def test_app_callback_routes_to_onboard_when_not_ready(monkeypatch):
    from cli import main as lingzhou_mod

    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(lingzhou_mod, "is_onboarded", lambda config: False)
    monkeypatch.setattr(
        lingzhou_mod,
        "onboard",
        lambda **kwargs: calls.append({"kind": "onboard", **kwargs}),
    )
    monkeypatch.setattr(
        lingzhou_mod,
        "gateway_start",
        lambda **kwargs: calls.append({"kind": "gateway", **kwargs}),
    )

    lingzhou_mod.app_callback(cast("Any", SimpleNamespace(invoked_subcommand=None)))

    assert calls == [{
        "kind": "onboard",
        "config": lingzhou_mod.DEFAULT_CONFIG_PATH,
        "start": True,
    }]


def test_app_callback_starts_gateway_when_ready(monkeypatch):
    from cli import main as lingzhou_mod

    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(lingzhou_mod, "is_onboarded", lambda config: True)
    monkeypatch.setattr(
        lingzhou_mod,
        "gateway_start",
        lambda **kwargs: calls.append(kwargs),
    )

    lingzhou_mod.app_callback(cast("Any", SimpleNamespace(invoked_subcommand=None)))

    assert calls == [{"channel": "local", "daemon": False}]


def test_gateway_startup_config_log_line_includes_requested_and_effective_paths():
    from cli.gateway import _startup_config_log_line

    cfg = cast(
        "Any",
        SimpleNamespace(
            _base_dir=Path("/tmp/runtime-home"),
            model="copilot/gpt-5.4",
            routing={
                "reader": "bailian/qwen3.6-plus",
                "reasoner": "copilot/gpt-5.4",
            },
        ),
    )

    line = _startup_config_log_line(
        cfg,
        Path("/tmp/requested/lingzhou.json"),
        channel="local",
        daemon=True,
    )

    assert "channel=local" in line
    assert "daemon=true" in line
    assert f"requested_config={Path('/tmp/requested/lingzhou.json').resolve()}" in line
    assert f"effective_config={(Path('/tmp/runtime-home') / 'lingzhou.json').resolve()}" in line
    assert "model_ref=copilot/gpt-5.4" in line
    assert "routing=reader=bailian/qwen3.6-plus, reasoner=copilot/gpt-5.4" in line


def test_runtime_config_snapshot_includes_effective_routing_summary():
    from core.loop.runtime.startup import _runtime_config_snapshot

    cfg = cast(
        "Any",
        SimpleNamespace(
            _base_dir=Path("/tmp/runtime-home"),
            model="copilot/gpt-5.4",
            routing={
                "reader": "bailian/qwen3.6-plus",
                "reasoner": "copilot/gpt-5.4",
            },
        ),
    )

    startup_line, routing_summary = _runtime_config_snapshot(
        cfg,
        {"reader": object()},
        stage="run",
    )

    assert "stage=run" in startup_line
    assert f"config={(Path('/tmp/runtime-home') / 'lingzhou.json').resolve()}" in startup_line
    assert "main_model=copilot/gpt-5.4" in startup_line
    assert "routing=reader=bailian/qwen3.6-plus, reasoner=copilot/gpt-5.4" in startup_line
    assert "reader: bailian/qwen3.6-plus ✓" in routing_summary
    assert "reasoner: copilot/gpt-5.4 (= main, no separate provider)" in routing_summary


def _config_doc_defaults(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        cols = [col.strip() for col in line.strip("|").split("|")]
        if len(cols) < 3:
            continue
        key = cols[0]
        if not (key.startswith("`") and key.endswith("`")):
            continue
        rows[key[1:-1]] = cols[1]
    return rows


def test_config_reference_doc_defaults_match_code_defaults():
    from core.config import config_reference_defaults

    doc_defaults = _config_doc_defaults(_proj_root() / "docs" / "reference" / "CONFIG.md")
    expected = config_reference_defaults()

    assert {key: doc_defaults.get(key) for key in expected} == expected


def test_example_config_matches_current_schema():
    from core.config import Config

    cfg = Config.load(_proj_root() / "lingzhou.json.example")

    assert cfg.memory.semantic_top_k == 5
    assert cfg.memory.daily_recall_days == 2
    assert cfg.soul.ethos.baseline.truth == 0.85


def test_memory_keys_in_thresholds_are_not_migrated():
    from core.config import Config

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.6-plus",
        "thresholds": {
            "daily_recall_days": 9,
            "daily_summary_days": 9,
        },
    })

    assert cfg.memory.daily_recall_days == 2
    assert cfg.memory.daily_summary_days == 7
    assert not hasattr(cfg.thresholds, "daily_recall_days")


def test_config_rejects_unknown_top_level_keys():
    from pydantic import ValidationError

    from core.config import Config

    with pytest.raises(ValidationError, match="unknown_section"):
        Config.model_validate({
            "providers": {
                "bailian": {
                    "type": "openai_compat",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "DASHSCOPE_API_KEY",
                }
            },
            "model": "bailian/qwen3.6-plus",
            "unknown_section": {},
        })


def test_config_defaults_provider_auth_profile_id_to_provider_name():
    from core.config import Config

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
                "base_url": "https://example.invalid",
                "api_key_env": "COPILOT_GITHUB_TOKEN",
            },
        },
        "model": "bailian/qwen3.6-plus",
    })

    assert cfg.providers["bailian"].auth_profile_id == "bailian:default"
    assert cfg.providers["copilot"].auth_profile_id == "copilot:default"


def test_create_provider_with_model_exposes_public_model_ref(monkeypatch):
    from core.config import Config
    from provider import create_provider_with_model
    from provider.base import EmbeddingProvider

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

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

    provider = create_provider_with_model(cfg, "bailian/qwen-plus")
    try:
        assert provider.model_ref == "bailian/qwen-plus"
        assert isinstance(provider, EmbeddingProvider)
    finally:
        asyncio.run(provider.close())


def test_gateway_start_prefers_config_default_channel_over_raw_json(monkeypatch, tmp_path):
    import core.loop as loop_mod
    from cli import gateway as gateway_mod

    requested_cfg = tmp_path / "lingzhou.json"
    requested_cfg.write_text('{"gateway": {"default_channel": "wechat"}}', encoding="utf-8")

    chosen_channels: list[str] = []
    cfg = cast(
        "Any",
        SimpleNamespace(
                gateway=SimpleNamespace(default_channel="local"),
                logging=SimpleNamespace(dir=str(tmp_path / "logs")),
                loop=SimpleNamespace(debug=False, act=True),
            _base_dir=tmp_path,
            model="copilot/gpt-5.4",
            routing={},
        ),
    )

    async def _noop_run() -> None:
        return None

    monkeypatch.setattr(gateway_mod, "onboarding_status", lambda config: (True, "ok"))
    monkeypatch.setattr(gateway_mod, "load_cfg", lambda config: cfg)
    monkeypatch.setattr(gateway_mod, "_is_systemd_managed", lambda: False)
    monkeypatch.setattr(gateway_mod, "_kill_existing_loop", lambda quiet=False: None)
    monkeypatch.setattr(gateway_mod, "_ensure_singleton", lambda: None)
    monkeypatch.setattr(gateway_mod, "_PID_FILE", tmp_path / "lingzhou.pid")
    monkeypatch.setattr(gateway_mod.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_mod,
        "_configure_lingzhou_logging",
        lambda log_dir, log_level, logger_name="lingzhou": (
            tmp_path / "lingzhou.log",
            tmp_path / "console.log",
        ),
    )
    monkeypatch.setattr(
        gateway_mod,
        "_startup_config_log_line",
        lambda cfg, requested_config, *, channel, daemon: chosen_channels.append(channel) or "line",
    )
    monkeypatch.setattr(loop_mod, "CognitionLoop", lambda cfg: SimpleNamespace(run=lambda: _noop_run()))

    gateway_mod.gateway_start(channel=None, config=requested_cfg, daemon=False)

    assert chosen_channels == ["local"]


def test_gateway_provider_preflight_reports_missing_active_key(monkeypatch):
    from cli import gateway as gateway_mod
    from core.config import Config
    from store import auth as auth_store

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(auth_store, "AUTH_PROFILES_PATH", Path("/tmp/lingzhou-missing-auth-profiles.json"))
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

    error = gateway_mod._gateway_provider_preflight_error(cfg)

    assert error is not None
    assert "provider 'bailian' 凭证不可用" in error
    assert "DASHSCOPE_API_KEY" in error


def test_gateway_provider_preflight_uses_default_auth_profile(monkeypatch, tmp_path):
    from cli import gateway as gateway_mod
    from core.config import Config
    from store import auth as auth_store

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(auth_store, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    auth_store.set_token_profile(profile_id="bailian:default", provider="bailian", token="sk-test-profile")
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

    assert cfg.providers["bailian"].auth_profile_id == "bailian:default"
    assert gateway_mod._gateway_provider_preflight_error(cfg) is None


def test_gateway_provider_preflight_does_not_require_fallback_provider_key(monkeypatch, tmp_path):
    from cli import gateway as gateway_mod
    from core.config import Config
    from store import auth as auth_store

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(auth_store, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    auth_store.set_token_profile(profile_id="bailian:default", provider="bailian", token="sk-test-profile")
    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            "deepseek": {
                "type": "openai_compat",
                "base_url": "https://deepseek.invalid/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
            },
        },
        "model": "bailian/qwen3.6-plus",
        "model_fallbacks": {
            "reasoner": ["deepseek/deepseek-chat"],
        },
    })

    assert gateway_mod._gateway_provider_preflight_error(cfg) is None


def test_gateway_provider_preflight_does_not_require_routing_provider_key(monkeypatch, tmp_path):
    from cli import gateway as gateway_mod
    from core.config import Config
    from store import auth as auth_store

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(auth_store, "AUTH_PROFILES_PATH", tmp_path / "auth-profiles.json")
    auth_store.set_token_profile(profile_id="bailian:default", provider="bailian", token="sk-test-profile")
    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            },
            "deepseek": {
                "type": "openai_compat",
                "base_url": "https://deepseek.invalid/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
            },
        },
        "model": "bailian/qwen3.6-plus",
        "routing": {
            "reasoner": "deepseek/deepseek-chat",
        },
    })

    assert gateway_mod._gateway_provider_preflight_error(cfg) is None


def test_gateway_start_stops_before_loop_when_provider_key_missing(monkeypatch, tmp_path):
    import core.loop as loop_mod
    from cli import gateway as gateway_mod
    from core.config import Config
    from store import auth as auth_store

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(auth_store, "AUTH_PROFILES_PATH", tmp_path / "missing-auth-profiles.json")
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

    printed: list[str] = []
    monkeypatch.setattr(gateway_mod, "onboarding_status", lambda config: (True, "ok"))
    monkeypatch.setattr(gateway_mod, "load_cfg", lambda config: cfg)
    monkeypatch.setattr(gateway_mod, "_is_systemd_managed", lambda: False)
    monkeypatch.setattr(gateway_mod, "_load_lingzhou_dotenv", lambda: None)
    monkeypatch.setattr(gateway_mod.console, "print", lambda *args, **kwargs: printed.append(str(args[0])))
    monkeypatch.setattr(loop_mod, "CognitionLoop", lambda cfg: (_ for _ in ()).throw(AssertionError("loop should not start")))

    with pytest.raises(typer.Exit):
        gateway_mod.gateway_start(channel="local", config=tmp_path / "lingzhou.json", daemon=False)

    assert any("Provider 凭证不可用" in line for line in printed)
    assert any("lingzhou dev doctor" in line for line in printed)


def test_enqueue_webhook_task_uses_ingress_store():
    from channels import webhook as webhook_mod

    calls: list[dict[str, str]] = []

    class _FakeIngress:
        def add_task(self, title: str, *, goal: str = "", priority: str = "normal", source: str = "external") -> int:
            calls.append({
                "title": title,
                "goal": goal,
                "priority": priority,
                "source": source,
            })
            return 7

    task_id = webhook_mod._enqueue_webhook_task(cast("Any", _FakeIngress()), "第一行\n第二行", "high")

    assert task_id == 7
    assert calls == [{
        "title": "webhook: 第一行 第二行",
        "goal": "第一行\n第二行",
        "priority": "high",
        "source": "gateway:webhook",
    }]


def test_normalize_webhook_message_merges_images_field():
    from channels import webhook as webhook_mod

    payload = {
        "message": "这是新任务",
        "images": [
            "img://a.png",
            {"path": "/tmp/b.png"},
            "  ",
            "file://c.png",
        ],
        "voices": ["voice://rec1.mp3"],
        "priority": "low",
    }

    msg, priority = webhook_mod._normalize_webhook_message(payload)

    assert priority == "low"
    assert msg == (
        "这是新任务\n[图片消息] img://a.png\n[图片消息] {\"path\": \"/tmp/b.png\"}\n[图片消息] file://c.png\n[语音消息] voice://rec1.mp3"
    )


def test_normalize_webhook_message_merges_audio_alias_field():
    from channels import webhook as webhook_mod

    payload = {
        "message": "legacy audio",
        "audios": "audio://legacy.amr",
        "priority": "normal",
    }

    msg, priority = webhook_mod._normalize_webhook_message(payload)

    assert priority == "normal"
    assert msg == "legacy audio\n[语音消息] audio://legacy.amr"


def test_start_external_channel_runtime_delegates_to_channel_registry(monkeypatch):
    from cli import gateway as gateway_mod

    printed: list[str] = []
    calls: list[tuple[Any, ...]] = []

    monkeypatch.setattr(gateway_mod.console, "print", lambda message, *args, **kwargs: printed.append(message))
    monkeypatch.setattr(
        gateway_mod,
        "describe_channel_runtime",
        lambda channel, gw_conf: calls.append(("describe", channel, dict(gw_conf))) or "runtime-line",
    )
    monkeypatch.setattr(
        gateway_mod,
        "start_channel_runtime",
        lambda channel, gw_conf, db_path: calls.append(("start", channel, dict(gw_conf), str(db_path))) or "runtime",
    )

    runtime = gateway_mod._start_external_channel_runtime(
        "wechat",
        {"token": "abc"},
        db_path="/tmp/lingzhou.db",
    )

    assert runtime == "runtime"
    assert printed == ["[dim]runtime-line[/dim]"]
    assert calls == [
        ("describe", "wechat", {"token": "abc"}),
        ("start", "wechat", {"token": "abc"}, "/tmp/lingzhou.db"),
    ]


def test_gateway_start_external_channel_waits_until_runtime_ready(monkeypatch, tmp_path):
    import core.loop as loop_mod
    from cli import gateway as gateway_mod

    requested_cfg = tmp_path / "lingzhou.json"
    requested_cfg.write_text('{"gateway": {"default_channel": "wechat"}}', encoding="utf-8")

    home_dir = tmp_path / "home"
    gw_dir = home_dir / ".lingzhou" / "gateway"
    gw_dir.mkdir(parents=True, exist_ok=True)
    (gw_dir / "wechat.json").write_text('{"channel": "wechat", "token": "abc"}', encoding="utf-8")

    cfg = cast(
        "Any",
        SimpleNamespace(
                gateway=SimpleNamespace(default_channel="wechat", webhook_host="0.0.0.0", webhook_port=8765),
                logging=SimpleNamespace(dir=str(tmp_path / "logs")),
                loop=SimpleNamespace(debug=False, act=True),
            _base_dir=tmp_path,
            model="copilot/gpt-5.4",
            routing={},
            db_path=tmp_path / "runtime.db",
        ),
    )

    events: list[str] = []

    class _FakeLoop:
        def __init__(self, _cfg):
            self._runtime_ready_callback = None

        async def run(self) -> None:
            events.append("run_enter")
            assert self._runtime_ready_callback is not None
            assert events == ["run_enter"]
            self._runtime_ready_callback()
            events.append("run_exit")

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(gateway_mod, "onboarding_status", lambda config: (True, "ok"))
    monkeypatch.setattr(gateway_mod, "load_cfg", lambda config: cfg)
    monkeypatch.setattr(gateway_mod, "_is_systemd_managed", lambda: False)
    monkeypatch.setattr(gateway_mod, "_kill_existing_loop", lambda quiet=False: None)
    monkeypatch.setattr(gateway_mod, "_ensure_singleton", lambda: None)
    monkeypatch.setattr(gateway_mod, "_PID_FILE", tmp_path / "lingzhou.pid")
    monkeypatch.setattr(gateway_mod.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_mod,
        "_configure_lingzhou_logging",
        lambda log_dir, log_level, logger_name="lingzhou": (
            tmp_path / "lingzhou.log",
            tmp_path / "console.log",
        ),
    )
    monkeypatch.setattr(gateway_mod, "_startup_config_log_line", lambda *args, **kwargs: "line")
    monkeypatch.setattr(
        gateway_mod,
        "_start_external_channel_runtime",
        lambda channel, gw_conf, db_path: events.append(f"channel_start:{channel}"),
    )
    monkeypatch.setattr(loop_mod, "CognitionLoop", _FakeLoop)

    gateway_mod.gateway_start(channel=None, config=requested_cfg, daemon=False)

    assert events == ["run_enter", "channel_start:wechat", "run_exit"]


@pytest.mark.asyncio
async def test_chat_interactive_assistant_reply_redraws_prompt_once(monkeypatch):
    from cli import chat as chat_mod

    class _FakeStore:
        def __init__(self):
            self._history_loaded = False
            self._reply_sent = False

        async def get_chat_messages_since(self, since: int, *, chat_id: str = ""):
            if since == 0 and not self._history_loaded:
                self._history_loaded = True
                return []
            if not self._reply_sent:
                self._reply_sent = True
                return [{"id": 1, "role": "assistant", "content": "爸爸，收到。"}]
            return []

        async def add_chat_message(self, role: str, content: str, *, chat_id: str = ""):
            return 2

    monkeypatch.setattr(chat_mod.console, "print", lambda *args, **kwargs: None)
    prompt_calls: list[str] = []
    monkeypatch.setattr(chat_mod, "_print_input_prompt", lambda prompt: prompt_calls.append(prompt))

    def _delayed_eof(prompt: str = "") -> str:
        time.sleep(0.4)
        return ""

    monkeypatch.setattr(chat_mod, "_read_line", _delayed_eof)

    await chat_mod._interactive(cast("Any", _FakeStore()), _test_config(), "", "灵舟")

    # _infer_user_title_from_messages 只看 role="user" 消息中的自报称谓；
    # 助手回复"爸爸，收到。"不触发 user_title 推断，退回到 "chat> "
    assert prompt_calls == ["chat> "]


def test_configure_lingzhou_logging_resets_console_log_each_time():
    from cli.gateway import _configure_lingzhou_logging

    with tempfile.TemporaryDirectory() as d:
        log_dir = Path(d)
        logger_name = "lingzhou.test.logging"
        logger = logging.getLogger(logger_name)
        old_handlers = list(logger.handlers)
        old_level = logger.level
        old_propagate = logger.propagate
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        try:
            _, console_log = _configure_lingzhou_logging(log_dir, logging.INFO, logger_name=logger_name)
            logger.info("first message")
            for handler in logger.handlers:
                handler.flush()

            _, console_log = _configure_lingzhou_logging(log_dir, logging.INFO, logger_name=logger_name)
            logger.info("second message")
            for handler in logger.handlers:
                handler.flush()

            text = console_log.read_text(encoding="utf-8")
            assert "second message" in text
            assert "first message" not in text
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            logger.handlers = old_handlers
            logger.setLevel(old_level)
            logger.propagate = old_propagate


def test_judgment_context_budget_trims_low_priority_sections():
    from core.judgment.context.budget import apply_context_budget

    ctx = {
        "task_section": "T" * 2000,
        "emotion_valence": "0.50",
        "emotion_arousal": "0.50",
        "emotion_dominant": "中性",
        "emotion_regulation": "stable",
        "wm_section": "W" * 1800,
        "failures_section": "F" * 800,
        "episodic_section": "E" * 2400,
        "memories_section": "M" * 2200,
        "soul_section": "S" * 900,
        "tools_section": "U" * 2000,
        "perception_section": "P" * 700,
        "ethos_section": "H" * 700,
        "signals_section": "G" * 700,
        "hard_boundaries_section": "B" * 700,
        "perception_replay_section": "R" * 700,
        "skills_section": "K" * 3000,
        "cognitive_signals_section": "C" * 700,
        "user_message": "",
    }

    budgeted = apply_context_budget(ctx, max_chars=12000)

    assert len(budgeted["task_section"]) == len(ctx["task_section"])
    assert len(budgeted["soul_section"]) == len(ctx["soul_section"])
    assert len(budgeted["skills_section"]) <= len(ctx["skills_section"])
    assert len(budgeted["memories_section"]) <= len(ctx["memories_section"])
    assert len(budgeted["episodic_section"]) <= len(ctx["episodic_section"])
    assert len(budgeted["wm_section"]) <= len(ctx["wm_section"])


def test_judgment_error_classification_and_cooldown():
    from core.config import Config
    from core.judgment import JudgmentLayer
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

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
    assert layer._executor._classify_error_code("Client error '429 Too Many Requests'") == "429"
    assert layer._executor._classify_error_code("Client error '400 Bad Request'") == "400"
    assert layer._executor._classify_error_code("ReadTimeout('')") == "timeout"

    assert layer._executor._cooldown_seconds("429", 1) >= 30
    assert layer._executor._cooldown_seconds("429", 3) > layer._executor._cooldown_seconds("429", 1)
    assert layer._executor._cooldown_seconds("400", 2) >= 90


def test_catalog_runtime_context_window_hint_is_effective():
    from provider.catalog import resolve_context_window, set_context_window_hint

    model_id = "adaptive-runtime-test-model"
    assert resolve_context_window(model_id, None) is None
    set_context_window_hint(model_id, 128000)
    assert resolve_context_window(model_id, None) == 128000
    assert resolve_context_window(model_id, 64000) == 64000


def test_judgment_executor_retries_with_trimmed_prompt_on_limit_error():
    asyncio.run(_judgment_executor_retries_with_trimmed_prompt_on_limit_error())


async def _judgment_executor_retries_with_trimmed_prompt_on_limit_error():
    from core.config import Config
    from core.judgment.executor import JudgmentExecutor
    from core.judgment.output import ModelSelection
    from provider.base import Message
    from provider.catalog import resolve_context_window

    class _Provider:
        model_ref = "copilot/adaptive-mini"

        def __init__(self) -> None:
            self.calls = 0
            self.lengths: list[int] = []

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            self.lengths.append(len(str(messages[-1].content)))
            if self.calls == 1:
                raise RuntimeError(
                    "Client error '400 Bad Request' for url 'https://api.individual.githubcopilot.com/responses' "
                    "body={\"error\":{\"message\":\"prompt token count of 153330 exceeds the limit of 128000\","
                    "\"code\":\"model_max_prompt_tokens_exceeded\"}}"
                )
            return '{"decision":"wait","rationale":"ok"}'

        async def close(self):
            return None

        async def ping(self, timeout: float = 8.0):
            return True, 1, None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "base_url": "https://api.individual.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            }
        },
        "model": "copilot/adaptive-mini",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    provider = _Provider()
    executor = JudgmentExecutor(provider, cfg)

    huge_context = "a" * 520000
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content=huge_context),
    ]
    selection = ModelSelection(phase="initial", tier="reasoner", model_ref="copilot/adaptive-mini", thinking="high")

    raw, final_selection, last_error = await executor._chat_with_retry(
        selected_provider=provider,
        selection=selection,
        messages=messages,
        phase="initial",
        user_message="hi",
        thinking_override="high",
        routing_overrides=None,
        log_prefix="[test]",
    )

    assert last_error is None
    assert final_selection.model_ref == "copilot/adaptive-mini"
    assert raw == '{"decision":"wait","rationale":"ok"}'
    assert provider.calls == 2
    assert provider.lengths[1] < provider.lengths[0]
    assert resolve_context_window("adaptive-mini", None) == 128000


def test_chat_with_retry_trims_before_first_call_when_budget_exceeded():
    asyncio.run(_chat_with_retry_trims_before_first_call_when_budget_exceeded())


async def _chat_with_retry_trims_before_first_call_when_budget_exceeded():
    from core.config import Config
    from core.judgment.executor import JudgmentExecutor
    from core.judgment.output import ModelSelection
    from provider.base import Message

    from provider.catalog import resolve_context_window

    class _Provider:
        model_ref = "copilot/adaptive-mini"

        def __init__(self) -> None:
            self.calls = 0
            self.total_lengths: list[int] = []

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            self.total_lengths.append(sum(len(str(message.content)) for message in messages))
            return '{"decision":"wait","rationale":"ok"}'

        async def close(self):
            return None

        async def ping(self, timeout: float = 8.0):
            return True, 1, None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "base_url": "https://api.individual.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            }
        },
        "model": "copilot/adaptive-mini",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    provider = _Provider()
    executor = JudgmentExecutor(provider, cfg)

    huge_user = "a" * 520000
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content=huge_user),
    ]
    selection = ModelSelection(phase="initial", tier="reasoner", model_ref="copilot/adaptive-mini", thinking="high")

    original_len = len(huge_user) + 3
    trim_calls: list[int] = []

    def _fake_trim(messages, prompt_limit, *, prompt_count=None):
        trim_calls.append(1)
        return [
            Message(role="system", content="sys"),
            Message(role="user", content="trimmed"),
        ]

    from unittest.mock import patch

    # 用 monkeypatch 记录是否触发了前置裁剪分支。
    with patch.object(executor, "_trim_messages_for_prompt_limit", _fake_trim):
        _, final_selection, last_error = await executor._chat_with_retry(
            selected_provider=provider,
            selection=selection,
            messages=messages,
            phase="initial",
            user_message="hi",
            thinking_override="high",
            routing_overrides=None,
            log_prefix="[test]",
            fallback_prefer_tier="reasoner",
            skills="none",
        )

    assert last_error is None
    assert final_selection.model_ref == "copilot/adaptive-mini"
    assert provider.calls == 1
    assert trim_calls == [1]
    assert provider.total_lengths[0] < original_len
    assert resolve_context_window("adaptive-mini", None) == 128000


def test_chat_with_retry_trims_largest_message_even_when_system_is_overlong():
    asyncio.run(_chat_with_retry_trims_largest_message_even_when_system_is_overlong())


async def _chat_with_retry_trims_largest_message_even_when_system_is_overlong():
    from core.config import Config
    from core.judgment.executor import JudgmentExecutor
    from core.judgment.output import ModelSelection
    from provider.base import Message

    class _Provider:
        model_ref = "copilot/adaptive-mini"

        def __init__(self) -> None:
            self.calls = 0
            self.total_lengths: list[int] = []

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            self.calls += 1
            self.total_lengths.append(sum(len(str(message.content)) for message in messages))
            if self.calls == 1:
                raise RuntimeError(
                    "Client error '400 Bad Request' for url 'https://api.individual.githubcopilot.com/responses' "
                    "body={\"error\":{\"message\":\"prompt token count of 153330 exceeds the limit of 128000\","
                    "\"code\":\"model_max_prompt_tokens_exceeded\"}}"
                )
            return '{"decision":"wait","rationale":"ok"}'

        async def close(self):
            return None

        async def ping(self, timeout: float = 8.0):
            return True, 1, None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "base_url": "https://api.individual.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            }
        },
        "model": "copilot/adaptive-mini",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    provider = _Provider()
    executor = JudgmentExecutor(provider, cfg)

    huge_system = "s" * 520000
    messages = [
        Message(role="system", content=huge_system),
        Message(role="user", content="hi"),
    ]
    selection = ModelSelection(phase="initial", tier="reasoner", model_ref="copilot/adaptive-mini", thinking="high")

    raw, final_selection, last_error = await executor._chat_with_retry(
        selected_provider=provider,
        selection=selection,
        messages=messages,
        phase="initial",
        user_message="hi",
        thinking_override="high",
        routing_overrides=None,
        log_prefix="[test]",
    )

    assert last_error is None
    assert final_selection.model_ref == "copilot/adaptive-mini"
    assert raw == '{"decision":"wait","rationale":"ok"}'
    assert provider.calls == 2
    assert provider.total_lengths[1] < provider.total_lengths[0]


def test_judgment_executor_logs_llm_usage(caplog):
    asyncio.run(_judgment_executor_logs_llm_usage(caplog))


async def _judgment_executor_logs_llm_usage(caplog):
    from core.config import Config
    from core.judgment.executor import JudgmentExecutor
    from core.judgment.output import ModelSelection
    from provider.base import Message

    class _Provider:
        model_ref = "copilot/adaptive-mini"

        def __init__(self) -> None:
            self.last_usage = {
                "prompt_tokens": 321,
                "completion_tokens": 45,
                "total_tokens": 366,
            }

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return '{"decision":"wait","rationale":"ok"}'

        async def close(self):
            return None

        async def ping(self, timeout: float = 8.0):
            return True, 1, None

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "base_url": "https://api.individual.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            }
        },
        "model": "copilot/adaptive-mini",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    provider = _Provider()
    executor = JudgmentExecutor(provider, cfg)
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello world"),
    ]
    selection = ModelSelection(phase="initial", tier="reasoner", model_ref="copilot/adaptive-mini", thinking="high")

    caplog.clear()
    caplog.set_level(logging.INFO, logger="lingzhou.judgment")
    raw, final_selection, last_error = await executor._chat_with_retry(
        selected_provider=provider,
        selection=selection,
        messages=messages,
        phase="initial",
        user_message="hi",
        thinking_override="high",
        routing_overrides=None,
        log_prefix="[test]",
        skills="memory.search,file.read",
    )

    assert raw == '{"decision":"wait","rationale":"ok"}'
    assert final_selection.model_ref == "copilot/adaptive-mini"
    assert last_error is None
    logs = [record.getMessage() for record in caplog.records if record.name == "lingzhou.judgment"]
    llm_logs = [msg for msg in logs if "[llm] ok" in msg]
    assert llm_logs
    assert "model_ref=copilot/adaptive-mini" in llm_logs[-1]
    assert "usage_prompt=321" in llm_logs[-1]
    assert "usage_completion=45" in llm_logs[-1]
    assert "usage_total=366" in llm_logs[-1]
    assert "usage_source=missing" in llm_logs[-1]
    assert "skills=memory.search,file.read" in llm_logs[-1]


def test_evolution_verification_outcome():
    from core.evolution import _verification_outcome

    baseline = {"runs": 4, "failures": 2, "successes": 1}
    assert _verification_outcome(baseline, {"runs": 1, "failures": 0, "successes": 1}, 3) == "pending"
    assert _verification_outcome(baseline, {"runs": 3, "failures": 0, "successes": 3}, 3) == "verified"
    assert _verification_outcome(baseline, {"runs": 3, "failures": 3, "successes": 0}, 3) == "regressed"


def test_exec_background_result_stays_running():
    from core.execution import _run_status_from_result
    from tools.registry import ToolResult

    result = ToolResult(
        summary="后台进程已启动",
        state_delta={"process": "started", "background": True},
        metadata={"session_id": "exec-1"},
    )
    assert _run_status_from_result(result) == "running"


def test_worker_layer_dispatches_specialized_handlers():
    asyncio.run(_worker_layer_dispatches_specialized_handlers())


async def _worker_layer_dispatches_specialized_handlers():
    from core.execution import WorkerLayer
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    async def _handler(params, ctx):
        return ToolResult(
            summary="ok",
            resource_key=str(params.get("resource_key") or ""),
            state_delta=dict(params.get("state_delta") or {}),
            metadata=dict(params.get("metadata") or {}),
        )

    entry = ToolEntry(manifest=ToolManifest(name="demo", description="demo"), handler=_handler)
    layer = WorkerLayer(_test_config())
    ctx = _tool_ctx()

    exec_result = await layer.dispatch(
        "exec-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="exec", params={
            "resource_key": "exec-1",
            "state_delta": {"process": "started", "background": True},
        }),
        ctx,
    )
    assert exec_result.metadata["worker_path"] == "exec"
    assert exec_result.metadata["execution_mode"] == "background"
    assert exec_result.metadata["session_id"] == "exec-1"

    multimodal_result = await layer.dispatch(
        "multimodal-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="image.analyze", params={"paths": ["a.png", "b.png"]}),
        ctx,
    )
    assert multimodal_result.metadata["worker_path"] == "multimodal"
    assert multimodal_result.metadata["modality"] == "image"
    assert multimodal_result.metadata["input_count"] == 2

    llm_result = await layer.dispatch(
        "llm-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="llm.simulated", params={"monitor_fact_key": "run:llm-1"}),
        ctx,
    )
    assert llm_result.metadata["worker_path"] == "llm"
    assert llm_result.metadata["reasoning_mode"] == "tool-mediated-llm"
    assert llm_result.metadata["run_monitor"]["key"] == "run:llm-1"


def test_worker_layer_throttles_same_pool_concurrency():
    asyncio.run(_worker_layer_throttles_same_pool_concurrency())


async def _worker_layer_throttles_same_pool_concurrency():
    from core.execution import WorkerLayer
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    started: list[str] = []
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def _handler(params, ctx):
        label = str(params.get("label") or "")
        started.append(label)
        if label == "first":
            first_started.set()
            await release.wait()
        return ToolResult(summary=f"ok-{label}")

    entry = ToolEntry(manifest=ToolManifest(name="demo.pool", description="demo"), handler=_handler)
    cfg = cast("Any", SimpleNamespace(loop=SimpleNamespace(
        max_tool_chain_workers=1,
        max_exec_workers=1,
        max_multimodal_workers=1,
        max_llm_workers=1,
    )))
    layer = WorkerLayer(cfg)
    ctx = _tool_ctx()

    first = asyncio.create_task(layer.dispatch(
        "tool-chain-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="demo.pool", params={"label": "first"}),
        ctx,
    ))
    await first_started.wait()

    second = asyncio.create_task(layer.dispatch(
        "tool-chain-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="demo.pool", params={"label": "second"}),
        ctx,
    ))
    await asyncio.sleep(0)

    assert started == ["first"]

    await asyncio.sleep(0.01)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert started == ["first", "second"]
    assert first_result.metadata["worker_limit"] == 1
    assert second_result.metadata["worker_limit"] == 1
    assert second_result.metadata["worker_wait_ms"] > 0
    assert second_result.metadata["worker_inflight"] == 1


def test_worker_layer_keeps_pools_independent():
    asyncio.run(_worker_layer_keeps_pools_independent())


async def _worker_layer_keeps_pools_independent():
    from core.execution import WorkerLayer
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    started: set[str] = set()
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def _handler(params, ctx):
        label = str(params.get("label") or "")
        started.add(label)
        if len(started) >= 2:
            both_started.set()
        await release.wait()
        return ToolResult(summary=f"ok-{label}")

    entry = ToolEntry(manifest=ToolManifest(name="demo.worker", description="demo"), handler=_handler)
    cfg = cast("Any", SimpleNamespace(loop=SimpleNamespace(
        max_tool_chain_workers=1,
        max_exec_workers=1,
        max_multimodal_workers=1,
        max_llm_workers=1,
    )))
    layer = WorkerLayer(cfg)
    ctx = _tool_ctx()

    tool_task = asyncio.create_task(layer.dispatch(
        "tool-chain-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="demo.worker", params={"label": "tool"}),
        ctx,
    ))
    llm_task = asyncio.create_task(layer.dispatch(
        "llm-worker",
        entry,
        _judgment_output(decision="act", chosen_action_id="demo.worker", params={"label": "llm", "monitor_fact_key": "run:demo"}),
        ctx,
    ))

    await asyncio.wait_for(both_started.wait(), timeout=0.2)
    assert started == {"tool", "llm"}

    release.set()
    tool_result, llm_result = await asyncio.gather(tool_task, llm_task)
    assert tool_result.metadata["worker_type"] == "tool-chain-worker"
    assert llm_result.metadata["worker_type"] == "llm-worker"
    assert llm_result.metadata["run_monitor"]["key"] == "run:demo"


def test_infer_run_profile_uses_explicit_registry_capabilities():
    from core.execution import _infer_run_profile
    from tools.registry import ToolEntry, ToolManifest, ToolResult

    async def _noop_handler(params, ctx):
        return ToolResult(summary="noop")

    class _Registry:
        def get(self, name: str):
            if name == "demo.exec":
                return ToolEntry(
                    manifest=ToolManifest(
                        name="demo.exec",
                        description="demo",
                        capabilities=("run_spawn",),
                    ),
                    handler=_noop_handler,
                )
            if name == "demo.vision":
                return ToolEntry(
                    manifest=ToolManifest(
                        name="demo.vision",
                        description="demo",
                        capabilities=("multimodal",),
                    ),
                    handler=_noop_handler,
                )
            return None

    assert _infer_run_profile("demo.exec") == ("tool_chain", "tool-chain-worker")
    assert _infer_run_profile("demo.exec", registry=cast("Any", _Registry())) == ("exec", "exec-worker")
    assert _infer_run_profile("demo.vision", registry=cast("Any", _Registry())) == ("multimodal", "multimodal-worker")
    assert _infer_run_profile("demo.exec", {"monitor_fact_key": "run:1"}, registry=cast("Any", _Registry())) == ("llm", "llm-worker")


def test_judgment_output_action_label_summarizes_parallel_and_delegate():
    from core.judgment import JudgmentOutput

    parallel = JudgmentOutput(
        decision="act",
        parallel_actions=[
            {"action_id": "file.read", "params": {}},
            {"action_id": "file.list", "params": {}},
        ],
    )
    delegated = JudgmentOutput(
        decision="act",
        delegate_tasks=[
            {"id": "alpha", "goal": "read alpha"},
            {"id": "beta", "goal": "read beta"},
        ],
    )

    assert parallel.action_label() == "parallel(2)[file.read, file.list]"
    assert delegated.action_label() == "delegate(2)[alpha, beta]"


def test_task_store_fact_listing_and_delete():
    asyncio.run(_task_store_fact_listing_and_delete())


async def _task_store_fact_listing_and_delete():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "facts.db")
        await store.open()
        await store.set_fact("evolution:verify:file.read", json.dumps({"ok": True}), scope="system")
        await store.set_fact("misc:key", "v", scope="general")

        facts = await store.list_facts(prefix="evolution:verify:")
        assert len(facts) == 1
        assert facts[0][0] == "evolution:verify:file.read"

        await store.delete_fact("evolution:verify:file.read")
        value, found = await store.get_fact("evolution:verify:file.read")
        assert not found
        assert value == ""
        await store.close()


def test_task_store_signal_lifecycle():
    asyncio.run(_task_store_signal_lifecycle())


async def _task_store_signal_lifecycle():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "signals.db")
        await store.open()

        once_id = await store.add_signal(
            "一次性提醒",
            "2000-01-01 00:00:00",
            payload={"note": "ping"},
        )
        repeat_id = await store.add_signal(
            "循环提醒",
            "2000-01-01 00:00:00",
            repeat_secs=60,
            payload={"source": "heartbeat"},
        )

        due = await store.due_signals()
        assert [sig["id"] for sig in due] == [once_id, repeat_id]

        await store.ack_signal(once_id)
        once_signal = await store.get_signal(once_id)
        assert once_signal is not None
        assert once_signal["status"] == "done"

        repeat_before = await store.get_signal(repeat_id)
        assert repeat_before is not None
        await store.ack_signal(repeat_id)
        repeat_after = await store.get_signal(repeat_id)
        assert repeat_after is not None
        assert repeat_after["status"] == "pending"
        assert repeat_after["run_at"] != repeat_before["run_at"]

        pending = await store.list_signals(limit=10)
        assert [sig["id"] for sig in pending] == [repeat_id]

        await store.cancel_signal(repeat_id)
        cancelled = await store.get_signal(repeat_id)
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"
        await store.close()


def test_evolution_pending_verification_becomes_verified():
    asyncio.run(_evolution_pending_verification_becomes_verified())


async def _evolution_pending_verification_becomes_verified():
    from types import SimpleNamespace
    from typing import cast

    from core.config import Config
    from core.evolution import EvolutionEngine, _verification_fact_key
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""

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
        "evolution": {
            "verify_min_runs": 2,
            "auto_rollback_on_regression": True,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        await store.set_fact(
            _verification_fact_key("file.read"),
            json.dumps({
                "target": "file.read",
                "tool_path": str(Path(d) / "file.py"),
                "backup_path": str(Path(d) / "file.py.bak"),
                "created_at": created_at,
                "baseline": {"runs": 3, "failures": 2, "successes": 1},
            }, ensure_ascii=False),
            scope="system",
        )
        await store.add_run(tool_name="file.read", status="succeeded")
        await store.add_run(tool_name="file.read", status="succeeded")

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
        results = await engine._process_pending_verifications(cast("Any", SimpleNamespace(task_store=store)))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].target == "verify:file.read"

        _, found = await store.get_fact(_verification_fact_key("file.read"))
        assert not found
        await store.close()


def test_evolution_regression_triggers_rollback():
    asyncio.run(_evolution_regression_triggers_rollback())


async def _evolution_regression_triggers_rollback():
    from types import SimpleNamespace
    from typing import cast

    from core.config import Config
    from core.evolution import EvolutionEngine, _verification_fact_key
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""

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
        "evolution": {
            "verify_min_runs": 2,
            "auto_rollback_on_regression": True,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        tool_path = Path(d) / "demo_tool.py"
        backup_path = Path(d) / "demo_tool.py.bak"
        tool_path.write_text("VALUE = 'new'\n", encoding="utf-8")
        backup_path.write_text("VALUE = 'old'\n", encoding="utf-8")
        created_at = (datetime.now(UTC) - timedelta(seconds=1)).replace(microsecond=0).isoformat()
        await store.set_fact(
            _verification_fact_key("demo.tool"),
            json.dumps({
                "target": "demo.tool",
                "tool_path": str(tool_path),
                "backup_path": str(backup_path),
                "created_at": created_at,
                "baseline": {"runs": 2, "failures": 1, "successes": 1},
            }, ensure_ascii=False),
            scope="system",
        )
        await store.add_run(tool_name="demo.tool", status="failed")
        await store.add_run(tool_name="demo.tool", status="failed")
        await store.record_failure("demo.tool", "still broken")
        await store.record_failure("demo.tool", "still broken again")

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
        results = await engine._process_pending_verifications(cast("Any", SimpleNamespace(task_store=store)))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].target == "rollback:demo.tool"
        assert tool_path.read_text(encoding="utf-8") == "VALUE = 'old'\n"

        _, found = await store.get_fact(_verification_fact_key("demo.tool"))
        assert not found
        await store.close()


def test_evolution_expired_breaker_fact_is_cleaned_on_check():
    asyncio.run(_evolution_expired_breaker_fact_is_cleaned_on_check())


async def _evolution_expired_breaker_fact_is_cleaned_on_check():
    from types import SimpleNamespace
    from typing import cast

    from core.config import Config
    from core.evolution import EvolutionEngine
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""

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

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.set_fact(
            "evolution:breaker:file.read",
            json.dumps({
                "target": "file.read",
                "failure_streak": 2,
                "cooldown_until": 0,
            }, ensure_ascii=False),
            scope="system",
        )

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
        is_open, remain, streak = await engine._is_target_breaker_cooling_down(
            cast("Any", SimpleNamespace(task_store=store)),
            "file.read",
        )
        assert is_open is False
        assert remain == 0
        assert streak == 2
        _, found = await store.get_fact("evolution:breaker:file.read")
        assert not found
        await store.close()


def test_evolution_verification_regression_updates_breaker_streak():
    asyncio.run(_evolution_verification_regression_updates_breaker_streak())


async def _evolution_verification_regression_updates_breaker_streak():
    from types import SimpleNamespace
    from typing import cast

    from core.config import Config
    from core.evolution import EvolutionEngine, _verification_fact_key
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""

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
        "evolution": {
            "verify_min_runs": 2,
            "auto_rollback_on_regression": False,
            "breaker_fail_threshold": 2,
        },
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        await store.set_fact(
            _verification_fact_key("demo.tool"),
            json.dumps({
                "target": "demo.tool",
                "tool_path": str(Path(d) / "demo_tool.py"),
                "backup_path": str(Path(d) / "demo_tool.py.bak"),
                "created_at": created_at,
                "baseline": {"runs": 2, "failures": 1, "successes": 1},
            }, ensure_ascii=False),
            scope="system",
        )
        await store.add_run(tool_name="demo.tool", status="failed")
        await store.add_run(tool_name="demo.tool", status="failed")
        await store.record_failure("demo.tool", "broken")
        await store.record_failure("demo.tool", "broken again")

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
        results = await engine._process_pending_verifications(cast("Any", SimpleNamespace(task_store=store)))
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].target == "verify:demo.tool"

        breaker_raw, found = await store.get_fact("evolution:breaker:demo.tool")
        assert found
        payload = json.loads(breaker_raw)
        assert payload["failure_streak"] >= 1
        _, verify_found = await store.get_fact(_verification_fact_key("demo.tool"))
        assert not verify_found
        await store.close()


def test_smoke_test_module_persists_failure_artifacts(tmp_path):
    from core.evolution import EvolutionEngine

    module_path = tmp_path / "demo_tool.py"
    err = EvolutionEngine._smoke_test_module(
        "raise RuntimeError('boom import')\n",
        module_path,
        tmp_path,
    )

    assert err is not None
    source_artifact = tmp_path / ".demo_tool.smoke-failed.py"
    log_artifact = tmp_path / ".demo_tool.smoke-failed.log"
    assert source_artifact.exists()
    assert log_artifact.exists()
    assert "boom import" in source_artifact.read_text(encoding="utf-8")
    log_text = log_artifact.read_text(encoding="utf-8")
    assert "RuntimeError: boom import" in log_text
    assert str(source_artifact) in err
    assert str(log_artifact) in err


def test_smoke_test_module_current_image_generate_passes():
    from core.evolution import EvolutionEngine

    module_path = _proj_root() / "tools" / "image_gen.py"
    src = module_path.read_text(encoding="utf-8")
    err = EvolutionEngine._smoke_test_module(src, module_path, _proj_root())
    assert err is None


def test_smoke_test_module_package_init_uses_package_name():
    from core.evolution import EvolutionEngine

    module_path = _proj_root() / "core" / "plugin" / "__init__.py"
    src = module_path.read_text(encoding="utf-8")
    err = EvolutionEngine._smoke_test_module(src, module_path, _proj_root())
    assert err is None


def test_registered_smoke_tests_are_not_import_only():
    from core.smoke_tests import SMOKE_TESTS

    violations: list[str] = []
    for rel_path, snippet in SMOKE_TESTS.items():
        if "assert True" in snippet:
            violations.append(f"{rel_path}: contains bare assert True")
        if "or True" in snippet:
            violations.append(f"{rel_path}: contains or True")
        if not snippet.strip():
            violations.append(f"{rel_path}: empty snippet")

    assert not violations, "\n".join(violations)


def test_smoke_failure_summary_uses_single_header_line():
    from core.evolution import _smoke_failure_summary

    text = (
        "smoke test failed | module=tools/image_gen.py | failed_log=/tmp/x.log\n\n"
        "returncode=1\n\n[stderr]\nTraceback..."
    )
    assert _smoke_failure_summary(text) == "smoke test failed | module=tools/image_gen.py | failed_log=/tmp/x.log"


def test_evolution_skill_targets_workspace_skill_file(tmp_path):
    asyncio.run(_evolution_skill_targets_workspace_skill_file(tmp_path))


async def _evolution_skill_targets_workspace_skill_file(tmp_path):
    from types import SimpleNamespace
    from typing import cast

    from core.config import Config
    from core.evolution import EvolutionEngine
    from core.skill import _seed_skills_dir, workspace_skill_file
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return """---
name: runtime-bootstrap
description: Workspace-evolved bootstrap skill.
---
只在 workspace 副本里演化这份 skill。
"""

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
            "workspace_dir": str(tmp_path / "workspace"),
        },
        "evolution": {
            "backup": True,
        },
    })

    engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())
    reload_calls: list[str] = []
    ctx = cast("Any", SimpleNamespace(judgment=SimpleNamespace(reload_skills=lambda: reload_calls.append("reloaded"))))

    seed_path = _seed_skills_dir() / "runtime-bootstrap" / "SKILL.md"
    seed_before = seed_path.read_text(encoding="utf-8")

    result = await engine.evolve_skill("runtime-bootstrap", "让 bootstrap 更贴近 runtime workspace。", ctx=ctx)

    target_path = workspace_skill_file(cfg.workspace_dir, "runtime-bootstrap")
    assert result.success is True
    assert result.target == "skill:runtime-bootstrap"
    assert target_path.exists()
    assert "Workspace-evolved bootstrap skill." in target_path.read_text(encoding="utf-8")
    assert seed_path.read_text(encoding="utf-8") == seed_before
    assert reload_calls == ["reloaded"]


def test_score_candidate_returns_positive():
    from core.evolution import _score_candidate

    # 基础代码应得正分
    code = "def f():\n    pass\n"
    score = _score_candidate(code)
    assert isinstance(score, int)
    assert score > 0


def test_score_candidate_rewards_exception_handling():
    from core.evolution import _score_candidate

    code_with_except = (
        "def f():\n"
        "    try:\n"
        "        return 1\n"
        "    except ValueError:\n"
        "        return 0\n"
    )
    code_without = "def f():\n    return 1\n"
    assert _score_candidate(code_with_except) > _score_candidate(code_without)


def test_evolution_config_competitive_candidates_default():
    from core.config_models import EvolutionConfig

    cfg = EvolutionConfig()
    assert cfg.competitive_candidates == 1


def test_evolution_config_competitive_candidates_custom():
    from core.config import Config

    cfg = Config.model_validate({
        "providers": {"b": {"type": "openai_compat", "base_url": "https://x.invalid/v1", "api_key_env": "K"}},
        "model": "b/qwen",
        "temperature": 0.7,
        "timeout": 30,
        "loop": {},
        "evolution": {"competitive_candidates": 3},
    })
    assert cfg.evolution.competitive_candidates == 3


def test_competitive_evolve_routes_based_on_config():
    """run() 应按 competitive_candidates 值分支到 competitive_evolve_tool 或 evolve_tool。"""
    asyncio.run(_competitive_evolve_routes_based_on_config())


async def _competitive_evolve_routes_based_on_config():
    import pathlib
    import tempfile
    import types as _types
    from unittest.mock import MagicMock

    from core.config import Config
    from core.evolution import EvolutionEngine, EvolutionResult
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    class _DummyProvider:
        model_ref = "dummy/provider"

        async def chat(self, messages, *, temperature=None, thinking_override=None):
            return ""
        async def close(self): pass

    cfg = Config.model_validate({
        "providers": {"b": {"type": "openai_compat", "base_url": "https://x.invalid/v1", "api_key_env": "K"}},
        "model": "b/qwen",
        "temperature": 0.7,
        "timeout": 30,
        "loop": {},
        "evolution": {"competitive_candidates": 2, "trigger_min_failures": 1},
    })

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(pathlib.Path(d) / "db.db")
        await store.open()
        await store.add_task("t", goal="g")
        await store.record_failure(kind="shell.run", summary="test failure", context="")

        engine = EvolutionEngine(cfg, _DummyProvider(), ToolRegistry())

        # 让 tool_path 存在
        fake_tool_path = pathlib.Path(d) / "shell.py"
        fake_tool_path.write_text("# stub", encoding="utf-8")
        engine._tools_dir = pathlib.Path(d)

        called: list[str] = []

        async def _fake_competitive(tool_name, tool_path, feedback, num_candidates=2, ctx=None):
            called.append("competitive")
            return EvolutionResult(success=True, target=tool_name)

        async def _fake_evolve(tool_name, tool_path, feedback, ctx=None):
            called.append("single")
            return EvolutionResult(success=True, target=tool_name)

        async def _fake_ethos(ctx):
            return EvolutionResult(success=False, target="ethos_baseline")

        async def _fake_verif(ctx):
            return []

        engine.competitive_evolve_tool = _fake_competitive  # type: ignore[method-assign]
        engine.evolve_tool = _fake_evolve  # type: ignore[method-assign]
        engine.evolve_ethos = _fake_ethos  # type: ignore[method-assign]
        engine._process_pending_verifications = _fake_verif  # type: ignore[method-assign]

        # registry.get() 需返回非 None，直接 mock
        mock_entry = MagicMock()
        engine._registry = MagicMock()
        engine._registry.get.return_value = mock_entry

        ctx_obj = _types.SimpleNamespace(task_store=store, config=cfg)
        await engine.run(ctx_obj)  # type: ignore[arg-type]
        await store.close()

    assert "competitive" in called, f"Expected competitive_evolve_tool called, got: {called}"


async def test_refresh_running_runs_updates_finished_exec_runs():
    import os
    import time

    from core.loop.runs.refresh import refresh_running_runs
    from store.task import TaskStore
    from tools.exec import _MANAGER, ProcessInfo

    with tempfile.TemporaryDirectory() as d:
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(Path(d) / "proc-state")
        _MANAGER.clear()
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("后台任务", goal="等进程结束")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="exec",
            worker_type="exec-worker",
            status="running",
            tool_name="exec",
            session_id="exec-test-1",
        )
        info = ProcessInfo(
            session_id="exec-test-1",
            command="echo hi",
            started_at=time.time(),
            finished=True,
            return_code=0,
            stdout="hi\n",
            background=True,
        )
        _MANAGER.register(info)
        _MANAGER.mark_finished("exec-test-1", 0)

        updates = await refresh_running_runs(store)
        assert updates
        assert updates[0]["status"] == "succeeded"

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "succeeded"
        assert run.progress.strip() == "hi"
        assert run.output_json["stdout"].strip() == "hi"

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.result_json["last_run_status"] == "succeeded"

        await store.close()


async def test_refresh_running_runs_updates_process_monitored_non_exec_runs():
    import os
    import time

    from core.loop.runs.refresh import refresh_running_runs
    from store.task import TaskStore
    from tools.exec import _MANAGER, ProcessInfo

    with tempfile.TemporaryDirectory() as d:
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(Path(d) / "proc-state")
        _MANAGER.clear()
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("统一监控任务", goal="通过 process monitor 刷新")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            status="running",
            tool_name="demo.process",
            output_json={"metadata": {"run_monitor": {"kind": "process", "session_id": "proc-unified-1"}}},
        )
        info = ProcessInfo(
            session_id="proc-unified-1",
            command="echo unified",
            started_at=time.time(),
            finished=True,
            return_code=0,
            stdout="unified\n",
            background=True,
        )
        _MANAGER.register(info)
        _MANAGER.mark_finished("proc-unified-1", 0)

        updates = await refresh_running_runs(store)
        assert updates
        assert updates[0]["status"] == "succeeded"

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "succeeded"
        assert run.progress.strip() == "unified"
        await store.close()


async def test_refresh_running_runs_crystallizes_progress():
    import os
    import time

    from core.loop.runs.refresh import refresh_running_runs
    from store.task import TaskStore
    from tools.exec import _MANAGER, ProcessInfo

    with tempfile.TemporaryDirectory() as d:
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(Path(d) / "proc-state")
        _MANAGER.clear()
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("长任务", goal="观察中间进度")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="exec",
            worker_type="exec-worker",
            status="running",
            tool_name="exec",
            session_id="exec-test-2",
        )
        info = ProcessInfo(
            session_id="exec-test-2",
            command="echo progress",
            started_at=time.time(),
            finished=False,
            stdout=(
                "phase-01 downloading artifacts\n"
                "phase-02 unpacking workspace\n"
                "phase-03 indexing files\n"
                "phase-04 building plan\n"
                "phase-05 verifying progress\n"
            ),
            background=True,
        )
        _MANAGER.register(info)

        updates = await refresh_running_runs(store)
        assert updates
        assert updates[0]["status"] == "running"
        assert updates[0]["crystal"]

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert "phase-05" in run.progress

        progress, found = await store.get_fact(f"task:{task_id}:progress")
        assert found
        assert "phase-05" in progress

        await store.close()


async def test_refresh_running_runs_updates_fact_monitored_non_exec_runs():
    from core.loop.runs.refresh import refresh_running_runs
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        task_id = await store.add_task("推理任务", goal="等待外部状态")
        await store.set_fact(
            "run:llm-1",
            json.dumps({"status": "running", "progress": "phase-1 reasoning"}, ensure_ascii=False),
            scope="task",
        )
        run_id = await store.add_run(
            task_id=task_id,
            run_type="llm",
            worker_type="llm-worker",
            status="running",
            tool_name="llm.simulated",
            output_json={
                "state_delta": {
                    "run_monitor": {
                        "kind": "fact",
                        "key": "run:llm-1",
                        "status_field": "status",
                        "progress_field": "progress",
                        "success_values": ["succeeded"],
                        "failed_values": ["failed"],
                    }
                }
            },
        )

        first = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert first
        assert first[0]["status"] == "running"
        assert "phase-1" in first[0]["crystal"]

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"
        assert "phase-1" in run.progress

        await store.set_fact(
            "run:llm-1",
            json.dumps({"status": "succeeded", "progress": "final answer ready", "summary": "done"}, ensure_ascii=False),
            scope="task",
        )
        second = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert second
        assert second[0]["status"] == "succeeded"

        finished = await store.get_run_by_id(run_id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.progress == "final answer ready"

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.result_json["last_run_status"] == "succeeded"

        completed = episodic.list_events("run_completed", limit=5)
        assert completed and completed[-1]["run_id"] == run_id

        node = semantic.get(f"run-result-{run_id}")
        assert node is not None
        assert node.kind == "run_result"

        await store.close()


async def test_refresh_running_runs_failed_fact_monitored_run_records_learning():
    from core.loop.runs.refresh import refresh_running_runs
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        task_id = await store.add_task("推理失败任务", goal="等待外部状态失败")
        await store.set_fact(
            "run:llm-fail-1",
            json.dumps({"status": "failed", "progress": "empty path", "error": "EmptyPath", "summary": "empty path"}, ensure_ascii=False),
            scope="task",
        )
        run_id = await store.add_run(
            task_id=task_id,
            run_type="llm",
            worker_type="llm-worker",
            status="running",
            tool_name="file.read",
            output_json={
                "state_delta": {
                    "run_monitor": {
                        "kind": "fact",
                        "key": "run:llm-fail-1",
                        "status_field": "status",
                        "progress_field": "progress",
                        "success_values": ["succeeded"],
                        "failed_values": ["failed"],
                    }
                }
            },
        )

        updates = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert updates
        assert updates[0]["status"] == "failed"

        failures = await store.list_failures(limit=5)
        assert failures
        assert failures[0].kind == "file.read"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].run_id == run_id
        assert reflections[0].target_kind == "task_split"

        double_loop = episodic.list_events("double_loop_reflection", limit=5)
        assert double_loop and double_loop[-1]["run_id"] == run_id

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        await store.close()


async def test_refresh_running_runs_failed_exec_run_records_learning():
    import time

    from core.loop.runs.refresh import refresh_running_runs
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.exec import _MANAGER, ProcessInfo

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        task_id = await store.add_task("exec 失败任务", goal="等待后台失败")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="exec",
            worker_type="exec-worker",
            status="running",
            tool_name="file.read",
            session_id="exec-failed-1",
        )

        _MANAGER.clear()
        info = ProcessInfo(
            session_id="exec-failed-1",
            command="cat missing",
            started_at=time.time() - 3,
            finished=True,
            return_code=1,
            stdout="",
            stderr="EmptyPath",
            error="EmptyPath",
            background=True,
        )
        _MANAGER.register(info)

        updates = await refresh_running_runs(store, episodic=episodic, semantic=semantic)
        assert updates
        assert updates[0]["status"] == "failed"

        failures = await store.list_failures(limit=5)
        assert failures
        assert failures[0].kind == "file.read"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].run_id == run_id
        assert reflections[0].target_kind == "task_split"

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        await store.close()


def test_catalog_resolve_context_window():
    """内置目录能按 model ID 自动查找 context_window，显式 override 优先。"""
    from provider.catalog import resolve_context_window

    # 已收录模型：自动查找
    assert resolve_context_window("qwen3.6-plus", None) == 1000000
    assert resolve_context_window("qwen3.5-plus", None) == 131072
    assert resolve_context_window("kimi-k2.5", None) == 262144
    assert resolve_context_window("gpt-5-mini", None) == 128000

    # 显式 override 优先于目录值
    assert resolve_context_window("qwen3.6-plus", 32768) == 32768

    # 未收录模型返回 None
    assert resolve_context_window("unknown-model-xyz", None) is None


def test_bailian_catalog_capabilities_are_curated():
    from provider.catalog import lookup_model

    qwen36 = lookup_model("qwen3.6-plus")
    assert qwen36 is not None
    assert qwen36["input"] == ["text", "image"]
    assert qwen36["capabilities"] == ["text_generation", "thinking", "vision"]

    coder_next = lookup_model("qwen3-coder-next")
    assert coder_next is not None
    assert coder_next["capabilities"] == ["text_generation"]
    assert "thinking" not in coder_next

    kimi = lookup_model("kimi-k2.5")
    assert kimi is not None
    assert kimi["input"] == ["text", "image"]
    assert kimi["capabilities"] == ["text_generation", "thinking", "vision"]


def test_catalog_explicit_path_isolated_from_global_runtime_path(tmp_path):
    import json as _json

    from provider import catalog as catalog_mod

    runtime_a = tmp_path / "workspace-a" / "models.json"
    runtime_a.parent.mkdir(parents=True, exist_ok=True)
    runtime_a.write_text(
        _json.dumps({
            "demo": {"models": [{"id": "alpha", "context_window": 111}]},
        }),
        encoding="utf-8",
    )

    runtime_b = tmp_path / "workspace-b" / "models.json"
    runtime_b.parent.mkdir(parents=True, exist_ok=True)
    runtime_b.write_text(
        _json.dumps({
            "demo": {"models": [{"id": "alpha", "context_window": 222}]},
        }),
        encoding="utf-8",
    )

    assert catalog_mod.resolve_context_window("alpha", None, catalog_path=runtime_a) == 111
    assert catalog_mod.resolve_context_window("alpha", None, catalog_path=runtime_b) == 222
    explicit = catalog_mod.lookup_model("alpha", catalog_path=runtime_b)
    assert explicit is not None
    assert explicit["context_window"] == 222


def test_catalog_budget_auto_lookup():
    """Config 不填 context_window_tokens 时，目录自动推断自适应工作集预算。"""
    from core.config import Config

    cfg = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.5-plus",  # 目录里 context_window=131072
        "temperature": 0.7,
        "timeout": 60.0,
    })
    assert cfg.judgment_input_token_budget() == 32768

    capped_budget = Config.model_validate({
        "providers": {
            "bailian": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "bailian/qwen3.5-plus",  # 目录里 context_window=131072
        "temperature": 0.7,
        "timeout": 60.0,
        "max_judgment_input_tokens": 16000,
    })
    assert capped_budget.judgment_input_token_budget() == 16000


def test_judgment_budget_is_derived_from_model_window():
    import pytest

    from core.config import Config

    # 未收录模型 + 显式 context_window_tokens → 正常计算
    cfg = Config.model_validate({
        "providers": {
            "custom": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "custom/demo",
        "context_window_tokens": 8000,
        "temperature": 0.7,
        "timeout": 60.0,
    })
    assert cfg.judgment_input_token_budget() == 6000

    # 未收录模型 + 无 context_window_tokens → fail loud，不静默降级
    unknown = Config.model_validate({
        "providers": {
            "custom": {
                "type": "openai_compat",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "DASHSCOPE_API_KEY",
            }
        },
        "model": "custom/demo",
        "temperature": 0.7,
        "timeout": 60.0,
    })
    with pytest.raises(ValueError, match="context_window_tokens"):
        unknown.judgment_input_token_budget()


def test_tool_registry():
    from tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.discover(_proj_root() / "tools")
    names = [m.name for m in reg.list_manifests()]
    assert "image.analyze" in names
    assert "shell.run" in names
    assert "shell.capabilities" in names
    assert "task.complete" in names
    assert "memory.add_wm" in names
    assert "memory.search" in names
    assert "file.list" in names
    assert "file.edit" in names
    assert "exec" in names
    assert "process.write" in names
    assert "skill.list" in names
    assert "skill.search" in names
    assert "skill.activate" in names


def test_file_list_and_memory_search():
    asyncio.run(_file_list_and_memory_search())


def test_image_source_helpers():
    from tools.image import (
        _collect_image_sources,
        _image_part_from_source,
        _resolve_multimodal_model_ref,
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        local = root / "sample.png"
        local.write_bytes(b"\x89PNG\r\n\x1a\nsmall-test-image")

        sources = _collect_image_sources({
            "path": str(local),
            "paths": json.dumps(["https://example.com/demo.jpg"]),
        })
        assert sources == [str(local), "https://example.com/demo.jpg"]

        local_part = _image_part_from_source(str(local), "auto")
        assert local_part["type"] == "image_url"
        assert local_part["image_url"]["url"].startswith("data:image/png;base64,")

        remote_part = _image_part_from_source("https://example.com/demo.jpg", "high")
        assert remote_part["image_url"]["url"] == "https://example.com/demo.jpg"
        assert remote_part["image_url"]["detail"] == "high"

        ctx = cast("Any", SimpleNamespace(config=SimpleNamespace(model="bailian/qwen3.6-plus", active_provider_name="bailian")))
        assert _resolve_multimodal_model_ref(ctx, capability="vision", input_modality="image") == "bailian/qwen3.6-plus"


def test_image_model_routing_falls_back_to_vision_model():
    from tools.image import _resolve_multimodal_model_ref

    ctx = cast("Any", SimpleNamespace(config=SimpleNamespace(model="deepseek/deepseek-v4-pro", active_provider_name="deepseek")))
    routed = _resolve_multimodal_model_ref(ctx, capability="vision", input_modality="image")
    assert routed != "deepseek/deepseek-v4-pro"
    assert routed == "bailian/qwen3.6-plus"


async def _file_list_and_memory_search():
    from pathlib import Path

    from store.semantic import MemoryNode, SemanticMemory
    from store.task import TaskStore
    from tools.file import file_list, file_read
    from tools.memory import memory_add_semantic, memory_search

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / 'a.txt').write_text('hello', encoding='utf-8')
        (root / 'sub').mkdir()
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            semantic = cast("Any", SemanticMemory(root))
            ctx = _tool_ctx(workspace_dir=str(root), task_store=store, semantic=semantic)
            listed = await file_list({'path': str(root)}, ctx)
            assert 'a.txt' in listed.summary
            assert 'sub/' in listed.summary

            read_file = await file_read({'path': str(root / 'a.txt')}, ctx)
            assert read_file.error is None
            assert read_file.summary == 'hello'

            read_dir = await file_read({'path': str(root)}, ctx)
            assert read_dir.error == 'NotAFile'

            read_empty = await file_read({'path': ''}, ctx)
            assert read_empty.error == 'EmptyPath'

            await memory_add_semantic({'title': 'bug fix note', 'body': 'reader tasks should use qwen3.6-plus', 'kind': 'fact'}, ctx)
            semantic.upsert(MemoryNode(
                id='task-note-1',
                kind='fact',
                title='legacy runtime primary carrier',
                body='/root/.legacy-runtime/memory/main.sqlite',
                tags=['task:33', 'path:/root/.legacy-runtime/memory'],
            ))
            found = await memory_search({'query': 'bug'}, ctx)
            assert 'bug fix note' in found.summary

            filtered = await memory_search({'query': 'legacy runtime', 'task_id': '33', 'path_prefix': '/root/.legacy-runtime/memory'}, ctx)
            assert 'legacy runtime primary carrier' in filtered.summary

            excluded = await memory_search({'query': 'legacy runtime', 'task_id': '34'}, ctx)
            assert excluded.skipped is True
        finally:
            await store.close()


def test_memory_add_semantic_disambiguates_duplicate_titles():
    asyncio.run(_memory_add_semantic_disambiguates_duplicate_titles())


async def _memory_add_semantic_disambiguates_duplicate_titles():
    from pathlib import Path

    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.memory import memory_add_semantic

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            semantic = cast("Any", SemanticMemory(root))
            ctx = _tool_ctx(workspace_dir=str(root), task_store=store, semantic=semantic)

            first = await memory_add_semantic(
                {'title': '最小合法JSON输出约束', 'body': 'rule body', 'kind': 'rule'},
                ctx,
            )
            second = await memory_add_semantic(
                {'title': '最小合法JSON输出约束', 'body': 'constraint body', 'kind': 'constraint'},
                ctx,
            )

            first_id = first.evidence.split('node_id=', 1)[1]
            second_id = second.evidence.split('node_id=', 1)[1]
            first_node = semantic.get(first_id)
            second_node = semantic.get(second_id)

            assert first_node is not None
            assert second_node is not None
            assert first_node.title == '最小合法JSON输出约束'
            assert second_node.title.startswith('最小合法JSON输出约束 [constraint:')
            assert first_node.title != second_node.title
        finally:
            await store.close()


def test_reflect_structural_disambiguates_duplicate_titles():
    asyncio.run(_reflect_structural_disambiguates_duplicate_titles())


async def _reflect_structural_disambiguates_duplicate_titles():
    from memory.working import WMItem, WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.memory import reflect_structural

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / 'runtime.db')
        await store.open()
        try:
            await store.add_task('结构反思任务', goal='reflect')
            semantic = cast("Any", SemanticMemory(root / 'semantic'))
            episodic = EpisodicMemory(root / 'episodic')
            wm = WorkingMemory(capacity=8)
            wm.add(WMItem(kind='observation', content='重复结构洞察来源', priority=0.8))
            ctx = _tool_ctx(task_store=store, episodic=episodic, semantic=semantic, wm=wm)

            first = await reflect_structural({'title': 'memory检索质量基线', 'insight': '洞察A'}, ctx)
            second = await reflect_structural({'title': 'memory检索质量基线', 'insight': '洞察B'}, ctx)

            first_id = first.evidence.split('node_id=', 1)[1]
            second_id = second.evidence.split('node_id=', 1)[1]
            first_node = semantic.get(first_id)
            second_node = semantic.get(second_id)

            assert first_node is not None
            assert second_node is not None
            assert first_node.title == 'memory检索质量基线'
            assert second_node.title.startswith('memory检索质量基线 [structural:')
        finally:
            await store.close()


def test_exec_process_write_pipe_roundtrip():
    asyncio.run(_exec_process_write_pipe_roundtrip())


async def _exec_process_write_pipe_roundtrip():
    import json

    from tools.exec import _MANAGER, exec_run, process_log, process_poll, process_write

    _MANAGER.clear()
    ctx = _tool_ctx()
    try:
        res = await exec_run({
            "command": "python3 -c \"import sys; print(sys.stdin.readline().strip())\"",
            "background": True,
            "timeout": 2,
        }, ctx)
        sid = json.loads(res.evidence)["process_id"]
        await process_write({"process_id": sid, "data": "hello\\n", "eof": True}, ctx)

        for _ in range(40):
            poll = await process_poll({"process_id": sid}, ctx)
            status = json.loads(poll.summary)
            if status["status"] == "finished":
                break
            await asyncio.sleep(0.05)

        log = await process_log({"process_id": sid, "offset": 0, "limit": 200}, ctx)
        assert "hello" in log.summary
    finally:
        _MANAGER.clear()


def test_exec_process_timeout_background():
    asyncio.run(_exec_process_timeout_background())


async def _exec_process_timeout_background():
    import json

    from tools.exec import _MANAGER, exec_run, process_poll

    _MANAGER.clear()
    ctx = _tool_ctx()
    try:
        res = await exec_run({
            "command": "python3 -c \"import time; time.sleep(5)\"",
            "background": True,
            "timeout": 0.2,
        }, ctx)
        sid = json.loads(res.evidence)["process_id"]

        timed_out = False
        for _ in range(60):
            poll = await process_poll({"process_id": sid}, ctx)
            status = json.loads(poll.summary)
            if status["status"] == "finished":
                timed_out = bool(status["timed_out"])
                break
            await asyncio.sleep(0.05)

        assert timed_out is True
    finally:
        _MANAGER.clear()


def test_exec_and_shell_explicit_no_output():
    asyncio.run(_exec_and_shell_explicit_no_output())


async def _exec_and_shell_explicit_no_output():
    from tools.exec import exec_run
    from tools.shell import shell_run

    ctx = _tool_ctx()

    exec_res = await exec_run({"command": "python3 -c \"pass\""}, ctx)
    assert exec_res.error is None
    assert "(无输出)" in exec_res.summary

    shell_res = await shell_run({"command": "python3 -c \"pass\""}, ctx)
    assert shell_res.error is None
    assert "(无输出)" in shell_res.summary
    expected_base = {"process": "finished", "exit_code": 0, "timed_out": False}
    assert expected_base.items() <= shell_res.state_delta.items()
    assert shell_res.metadata["log_summary"].startswith("shell.run exit=0 chars=0")
    assert shell_res.resource_key is not None
    json.dumps(shell_res.to_dict(), ensure_ascii=False)


def test_probe_run_auto_rerun_when_name_missing():
    asyncio.run(_probe_run_auto_rerun_when_name_missing())


async def _probe_run_auto_rerun_when_name_missing():
    from core.contracts.probe import ProbeConfig, ProbeResult
    from tools.probe import probe_run

    class _FakeProbeMgr:
        async def list_probes(self):
            return [
                ProbeConfig(name="network_health", kind="shell", spec="echo 1", trigger="interval:60", enabled=True, last_error="timeout"),
                ProbeConfig(name="cpu_health", kind="shell", spec="echo 2", trigger="interval:60", enabled=True, last_confidence=0.4),
                ProbeConfig(name="ok_probe", kind="shell", spec="echo 3", trigger="interval:60", enabled=True, last_confidence=0.9),
            ]

        async def run_now(self, name: str):
            return ProbeResult(
                probe_name=name,
                output="ok",
                error=None,
                triggered_at="2026-05-31T00:00:00+00:00",
                duration_ms=5,
                confidence=0.85,
                confidence_reason="normal",
                deployment_suspect=False,
            )

    ctx = _tool_ctx()
    ctx.probe_manager = _FakeProbeMgr()  # type: ignore[attr-defined]

    res = await probe_run({}, ctx)
    assert res.error is None
    assert "name 为空，自动重跑" in res.summary
    assert "network_health" in res.summary


def test_shell_run_uses_bash_compatible_syntax():
    asyncio.run(_shell_run_uses_bash_compatible_syntax())


async def _shell_run_uses_bash_compatible_syntax():
    from tools.shell import shell_run

    ctx = _tool_ctx()
    # (( )) 是 bash 语法，/bin/sh 下常见会报 "Syntax error: ( unexpected"
    res = await shell_run({"command": "x=1; ((x++)); echo $x"}, ctx)
    assert res.error is None
    assert "2" in res.summary


def test_shell_run_summary_keeps_output_preview_only():
    asyncio.run(_shell_run_summary_keeps_output_preview_only())


async def _shell_run_summary_keeps_output_preview_only():
    import json

    from tools.shell import shell_run

    ctx = _tool_ctx()
    payload_cmd = (
        "python3 - <<'PY'\n"
        "import sys\n"
        "sys.stdout.write('A' * 5000)\n"
        "sys.stdout.write('\\n' + 'B' * 5000)\n"
        "PY"
    )
    res = await shell_run({"command": payload_cmd}, ctx)

    assert res.error is None
    assert len(res.summary) < 5000
    assert "omitted" in json.loads(res.evidence)["output_preview"]
    assert "output_chars" in json.loads(res.evidence)


def test_execution_durable_failure_sensing():
    asyncio.run(_execution_durable_failure_sensing())


async def _execution_durable_failure_sensing():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="exec",
            params={"command": "bash /definitely/missing-lingzhou-script.sh"},
            rationale="test durable failure sensing",
        )

        first = second = third = fourth = None
        first = await layer.dispatch(action, ctx)
        second = await layer.dispatch(action, ctx)
        third = await layer.dispatch(action, ctx)
        fourth = await layer.dispatch(action, ctx)

        assert first is not None and first.error
        assert second is not None and second.error
        assert third is not None and third.error
        assert fourth is not None
        assert fourth.skipped is True
        assert fourth.error == "KnownStableFailure"
        assert "跳过已知稳定失败动作" in fourth.summary

        await store.close()


def test_execution_durable_failure_sensing_for_file_tool():
    asyncio.run(_execution_durable_failure_sensing_for_file_tool())


async def _execution_durable_failure_sensing_for_file_tool():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": ""},
            rationale="test durable failure sensing for file tool",
        )

        for _ in range(3):
            res = await layer.dispatch(action, ctx)
            assert res.error == "EmptyPath"
        fourth = await layer.dispatch(action, ctx)
        assert fourth.skipped is True
        assert fourth.error == "KnownStableFailure"

        await store.close()


def test_classify_durable_failure_tolerates_non_string_tool_result_fields():
    from core.execution.helpers import _classify_durable_failure
    from tools.registry import ToolResult

    result = ToolResult(
        summary=cast("Any", {"command": "bash /definitely/missing.sh"}),
        evidence=cast("Any", {"stderr": "No such file or directory"}),
        error="CommandFailed",
    )

    assert _classify_durable_failure(result) == "missing_path"


def test_execution_dispatch_normalizes_non_string_tool_result_fields():
    asyncio.run(_execution_dispatch_normalizes_non_string_tool_result_fields())


async def _execution_dispatch_normalizes_non_string_tool_result_fields():
    from tools.registry import ToolManifest, ToolParam, ToolRegistry, ToolResult, tool

    tool_name = "probe.non_string_result"

    @tool(ToolManifest(
        name=tool_name,
        description="return malformed tool result",
        params=[ToolParam("value", "string", "dummy", required=False)],
    ))
    async def _malformed_result(params, ctx):
        return ToolResult(
            summary=cast("Any", {"command": "bash /definitely/missing.sh"}),
            evidence=cast("Any", {"stderr": "No such file or directory"}),
            error="CommandFailed",
        )

    reg = ToolRegistry()
    layer = _execution_layer(reg)
    ctx = _tool_ctx(debug=False, task_store=None)
    action = _judgment_output(decision="act", chosen_action_id=tool_name, params={})

    result = await layer.dispatch(action, ctx)

    assert isinstance(result.summary, str)
    assert isinstance(result.evidence, str)
    assert "definitely/missing.sh" in result.summary
    assert "No such file or directory" in result.evidence


def test_execution_dispatch_accepts_dict_result_on_llm_worker():
    asyncio.run(_execution_dispatch_accepts_dict_result_on_llm_worker())


async def _execution_dispatch_accepts_dict_result_on_llm_worker():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolManifest, ToolParam, ToolRegistry, tool

    tool_name = "probe.llm_dict_result"

    @tool(ToolManifest(
        name=tool_name,
        description="return dict result on llm worker",
        params=[
            ToolParam("monitor_fact_key", "string", "monitor key", required=False),
            ToolParam("value", "string", "dummy", required=False),
        ],
    ))
    async def _llm_dict_result(params, ctx):
        return cast("Any", {
            "summary": "llm worker dict ok",
            "evidence": {"status": "running"},
            "metadata": {"from": "dict"},
        })

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            reg = ToolRegistry()
            layer = _execution_layer(reg)
            ctx = _tool_ctx(debug=False, task_store=store)
            action = _judgment_output(
                decision="act",
                chosen_action_id=tool_name,
                params={"monitor_fact_key": "monitor:test"},
            )

            result = await layer.dispatch(action, ctx)

            assert result.summary == "llm worker dict ok"
            assert result.metadata["worker_path"] == "llm"
            assert result.metadata["reasoning_mode"] == "tool-mediated-llm"
            assert result.metadata["from"] == "dict"
        finally:
            await store.close()


def test_execution_dispatch_records_run():
    asyncio.run(_execution_dispatch_records_run())


async def _execution_dispatch_records_run():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        target = root / "demo.txt"
        target.write_text("hello", encoding="utf-8")
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        await store.add_task("读取文件", goal="读 demo", model_tier="reader")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": str(target)},
            rationale="record run",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error is None
        assert result.metadata.get("run_id")
        assert result.metadata.get("worker_type") == "tool-chain-worker"

        runs = await store.list_runs(limit=5)
        assert runs
        assert runs[0].status == "succeeded"
        assert runs[0].tool_name == "file.read"
        assert runs[0].model_tier == "reader"
        assert "hello" in runs[0].progress
        assert runs[0].output_json["summary"] == "hello"

        started = episodic.list_events("run_started", limit=5)
        completed = episodic.list_events("run_completed", limit=5)
        assert started and started[-1]["run_id"] == runs[0].id
        assert completed and completed[-1]["run_id"] == runs[0].id

        node = semantic.get(f"run-result-{runs[0].id}")
        assert node is not None
        assert node.kind == "run_result"
        assert "succeeded" in node.tags

        active = await store.get_active()
        assert active is not None
        assert active.result_json["last_run_id"] == runs[0].id
        assert active.result_json["last_run_status"] == "succeeded"
        assert active.result_json["worker_type"] == "tool-chain-worker"

        await store.close()


def test_record_run_outcome_memory_clips_large_semantic_body():
    asyncio.run(_record_run_outcome_memory_clips_large_semantic_body())


async def _record_run_outcome_memory_clips_large_semantic_body():
    from core.execution.helpers import record_run_outcome_memory
    from store.semantic import SemanticMemory

    with tempfile.TemporaryDirectory() as d:
        semantic = SemanticMemory(Path(d) / "memory")
        huge_summary = "A" * 7000 + "TAIL"

        await record_run_outcome_memory(
            episodic=None,
            semantic=semantic,
            memory_cfg=None,
            run_id=991,
            task_id=12,
            tool_name="shell.run",
            worker_type="tool-chain-worker",
            status="succeeded",
            progress="ok",
            summary=huge_summary,
            error="",
        )

        node = semantic.get("run-result-991")
        assert node is not None
        assert len(node.body) < len(huge_summary)
        assert "run_result memory truncated" in node.body
        assert "TAIL" in node.body


def test_execution_logs_worker_metadata(caplog):
    asyncio.run(_execution_logs_worker_metadata(caplog))


async def _execution_logs_worker_metadata(caplog):
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        target = root / "demo.txt"
        target.write_text("hello", encoding="utf-8")
        store = TaskStore(root / "runtime.db")
        await store.open()
        await store.add_task("读取文件", goal="读 demo", model_tier="reader")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": str(target)},
            rationale="log worker metadata",
        )

        caplog.clear()
        caplog.set_level(logging.INFO, logger="lingzhou.execution")
        result = await layer.dispatch(action, ctx)

        assert result.error is None
        messages = [record.getMessage() for record in caplog.records if record.name == "lingzhou.execution"]
        tool_logs = [msg for msg in messages if "[tool-result]" in msg]
        finalize_logs = [msg for msg in messages if "[run-finalize]" in msg]
        assert tool_logs and "worker_meta=path=tool-chain" in tool_logs[-1]
        assert "limit=" in tool_logs[-1]
        assert "dispatch_ms=" in tool_logs[-1]
        assert finalize_logs and "worker_meta=path=tool-chain" in finalize_logs[-1]
        assert "dispatch_ms=" in finalize_logs[-1]

        await store.close()


def test_execution_dispatch_rebinds_target_task_run():
    asyncio.run(_execution_dispatch_rebinds_target_task_run())


async def _execution_dispatch_rebinds_target_task_run():
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        active_id = await store.add_task(
            "active task",
            goal="should keep focus but not own explicit target run",
            status="in_progress",
        )
        target_id = await store.add_task(
            "pending target",
            goal="should receive explicit task.advance run",
            status="pending",
        )
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="task.advance",
            params={"task_id": target_id},
            rationale="rebind explicit task target",
        )

        result = await layer.dispatch(action, ctx)

        assert result.error is None
        assert result.skipped is False
        runs = await store.list_runs(limit=5)
        assert runs
        assert runs[0].tool_name == "task.advance"
        assert runs[0].task_id == target_id

        target = await store.get_task_by_id(target_id)
        assert target is not None
        assert target.status == "in_progress"
        assert target.result_json["last_run_id"] == runs[0].id
        assert target.result_json["last_run_status"] == "succeeded"

        active = await store.get_task_by_id(active_id)
        assert active is not None
        assert active.status == "in_progress"
        assert active.result_json == {}

        await store.close()


def test_execution_dispatch_routes_fact_monitored_action_to_llm_worker():
    asyncio.run(_execution_dispatch_routes_fact_monitored_action_to_llm_worker())


async def _execution_dispatch_routes_fact_monitored_action_to_llm_worker():
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        await store.add_task("llm worker 任务", goal="验证 llm worker 选路")
        await store.set_fact("run:llm-route", json.dumps({"status": "running"}, ensure_ascii=False), scope="task")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="memory.get_fact",
            params={"key": "run:llm-route", "monitor_fact_key": "run:llm-route"},
            rationale="route to llm worker",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error is None
        assert result.metadata["worker_type"] == "llm-worker"
        assert result.metadata["run_monitor"]["kind"] == "fact"

        runs = await store.list_runs(limit=5)
        assert runs
        assert runs[0].run_type == "llm"
        assert runs[0].worker_type == "llm-worker"
        await store.close()


def test_execution_plan_gate_blocks_mutation_until_current_step_aligned():
    asyncio.run(_execution_plan_gate_blocks_mutation_until_current_step_aligned())


async def _execution_plan_gate_blocks_mutation_until_current_step_aligned():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("修复回复链路", goal="验证 plan gate")
        await store.update_task_data(task_id, {
            "plan": [
                {"step": "先搜索证据", "status": "in_progress"},
                {"step": "再写入修复", "status": "pending"},
            ]
        })

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)

        blocked = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="memory.set_fact",
                params={"key": "debug:test", "value": "1"},
                rationale="直接写入结论",
            ),
            ctx,
        )
        assert blocked.skipped is True
        assert blocked.error == "PlanStepMismatch"
        assert "先搜索证据" in blocked.summary
        value, found = await store.get_fact("debug:test")
        assert found is False
        assert value == ""

        align = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="task.update",
                params={"current_step": "先搜索证据"},
                rationale="先对齐当前步骤",
            ),
            ctx,
        )
        assert align.error is None

        allowed = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="memory.set_fact",
                params={"key": "debug:test", "value": "1"},
                rationale="步骤已对齐，允许写入",
            ),
            ctx,
        )
        assert allowed.error is None
        value, found = await store.get_fact("debug:test")
        assert found is True
        assert value == "1"

        await store.close()


def test_execution_plan_gate_keeps_reader_tools_available():
    asyncio.run(_execution_plan_gate_keeps_reader_tools_available())


def test_execution_dispatch_binds_run_to_ctx_focus_task():
    asyncio.run(_execution_dispatch_binds_run_to_ctx_focus_task())


async def _execution_plan_gate_keeps_reader_tools_available():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("分析问题", goal="验证 reader 不被 plan gate 误伤")
        await store.update_task_data(task_id, {
            "plan": [
                {"step": "先搜索证据", "status": "in_progress"},
                {"step": "再整理结论", "status": "pending"},
            ]
        })

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)

        result = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="task.list",
                params={"status": "all", "limit": 5},
                rationale="先看当前任务列表",
            ),
            ctx,
        )
        assert result.error is None
        assert "分析问题" in result.summary

        await store.close()


async def _execution_dispatch_binds_run_to_ctx_focus_task():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        global_active_id = await store.add_task(
            "全局活跃任务",
            goal="不应抢走 run 归属",
            status="in_progress",
        )
        focus_task_id = await store.add_task(
            "当前焦点任务",
            goal="run 应绑定这里",
            status="pending",
        )
        focus_task = await store.get_task_by_id(focus_task_id)
        assert focus_task is not None

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, active_task=focus_task)

        result = await layer.dispatch(
            _judgment_output(
                decision="act",
                chosen_action_id="memory.set_fact",
                params={"key": "focus:run-binding", "value": "ok"},
                rationale="验证 run 归属跟随当前 focus task",
            ),
            ctx,
        )

        run_id = int(result.metadata.get("run_id") or 0)
        run = await store.get_run_by_id(run_id)
        global_active = await store.get_task_by_id(global_active_id)

        assert run is not None
        assert run.task_id == focus_task_id
        assert global_active is not None and global_active.status == "in_progress"
        await store.close()


def test_execution_failure_creates_meta_reflection():
    asyncio.run(_execution_failure_creates_meta_reflection())


async def _execution_failure_creates_meta_reflection():
    from tempfile import TemporaryDirectory

    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        await store.add_task("读空路径", goal="制造失败")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": ""},
            rationale="trigger meta reflection",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error == "EmptyPath"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].target_kind == "task_split"
        assert reflections[0].loop_level == "double"
        assert reflections[0].tool_name == "file.read"
        assert reflections[0].decision == "apply"

        runs = await store.list_runs(limit=5)
        assert runs and runs[0].status == "cancelled"

        started = episodic.list_events("run_started", limit=5)
        failed = episodic.list_events("run_failed", limit=5)
        double_loop = episodic.list_events("double_loop_reflection", limit=5)
        assert started and started[-1]["run_id"] == runs[0].id
        assert failed and failed[-1]["run_id"] == runs[0].id
        assert double_loop and double_loop[-1]["run_id"] == runs[0].id

        node = semantic.get(f"run-result-{runs[0].id}")
        assert node is not None
        assert node.kind == "run_result"
        assert "failed" in node.tags

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        rule_node = semantic.get(f"rule-revision-{reflections[0].id}")
        assert rule_node is not None
        assert rule_node.kind == "rule_revision"
        assert rule_node.title == f"[apply] task_split via file.read run#{runs[0].id}"

        await store.close()


def test_execution_generic_failure_meta_reflection_defers():
    asyncio.run(_execution_generic_failure_meta_reflection_defers())


async def _execution_generic_failure_meta_reflection_defers():
    from tempfile import TemporaryDirectory

    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        await store.add_task("读缺失文件", goal="制造单环失败")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": str(root / "missing.txt")},
            rationale="trigger generic meta reflection",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error == "FileNotFound"

        reflections = await store.list_meta_reflections(limit=5)
        assert reflections
        assert reflections[0].target_kind == "tool"
        assert reflections[0].loop_level == "single"
        assert reflections[0].decision == "defer"

        double_loop = episodic.list_events("double_loop_reflection", limit=5)
        assert double_loop == []

        meta_node = semantic.get(f"meta-reflection-{reflections[0].id}")
        assert meta_node is not None
        assert meta_node.kind == "meta_reflection"

        rule_node = semantic.get(f"rule-revision-{reflections[0].id}")
        assert rule_node is None

        await store.close()


def test_ingest_actionable_meta_reflections_dedupes():
    asyncio.run(_ingest_actionable_meta_reflections_dedupes())


async def _ingest_actionable_meta_reflections_dedupes():
    from core.loop.task.runtime import _ingest_actionable_meta_reflections
    from memory.working import WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-apply",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="动作选择漂移",
            proposal="修正路由",
            verification_plan="重跑 judgment",
            decision="apply",
            task_id=7,
            run_id=3,
            tool_name="file.read",
        )
        await store.add_meta_reflection(
            reflection_id="mr-rollback",
            target_kind="threshold",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="阈值退化",
            proposal="回滚阈值",
            verification_plan="等待窗口结束",
            decision="rollback",
            task_id=8,
            run_id=4,
            tool_name="exec",
        )
        await store.add_meta_reflection(
            reflection_id="mr-defer",
            target_kind="tool",
            trigger="failure_pattern",
            loop_level="single",
            diagnosis="普通失败",
            proposal="稍后再看",
            verification_plan="重试",
            decision="defer",
            task_id=9,
            run_id=5,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-apply", "mr-rollback"]

        top = wm.get_top()
        assert len([item for item in top if item["kind"] == "meta_reflection"]) == 2
        assert any("queued task routing guard" in item["content"] for item in top)
        assert any("control:durable_failure_policy" in item["content"] for item in top)
        assert any("memory.set_fact" in item["content"] for item in top)

        task_fact, found = await store.get_fact("task:7:meta_reflection")
        assert found
        assert json.loads(task_fact)["decision"] == "apply"

        routing_guard, found = await store.get_fact("task:7:routing_guard")
        assert found
        assert json.loads(routing_guard)["tool_name"] == "file.read"

        policy_raw, found = await store.get_fact("control:durable_failure_policy")
        assert not found
        assert policy_raw == ""

        threshold_hint_raw, found = await store.get_fact("control:meta_reflection_hint:threshold")
        assert found
        assert json.loads(threshold_hint_raw)["suggested_policy"] == {"threshold": 3, "ttl_sec": 7200}

        again = await _ingest_actionable_meta_reflections(store, wm)
        assert again == []
        await store.close()


def test_meta_reflection_threshold_apply_surfaces_runtime_policy_hint():
    asyncio.run(_meta_reflection_threshold_apply_surfaces_runtime_policy_hint())


async def _meta_reflection_threshold_apply_surfaces_runtime_policy_hint():
    from core.loop.task.runtime import _ingest_actionable_meta_reflections
    from memory.working import WorkingMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-threshold-apply",
            target_kind="threshold",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="稳定失败阈值过紧",
            proposal="放宽 durable failure 阈值",
            verification_plan="第四次失败前不应静默",
            decision="apply",
            tool_name="file.read",
        )
        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-threshold-apply"]

        top = wm.get_top()
        assert any(
            item["kind"] == "meta_reflection"
            and "control:durable_failure_policy" in item["content"]
            and "memory.set_fact" in item["content"]
            for item in top
        )

        hint_raw, found = await store.get_fact("control:meta_reflection_hint:threshold")
        assert found
        assert json.loads(hint_raw)["suggested_policy"] == {"threshold": 4, "ttl_sec": 3600}

        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store)
        action = _judgment_output(
            decision="act",
            chosen_action_id="file.read",
            params={"path": ""},
            rationale="test threshold policy apply",
        )

        first = await layer.dispatch(action, ctx)
        second = await layer.dispatch(action, ctx)
        third = await layer.dispatch(action, ctx)
        fourth = await layer.dispatch(action, ctx)
        fifth = await layer.dispatch(action, ctx)

        assert first.error == "EmptyPath"
        assert second.error == "EmptyPath"
        assert third.error == "EmptyPath"
        assert fourth.error == "KnownStableFailure"
        assert fifth.error == "KnownStableFailure"

        policy_raw, found = await store.get_fact("control:durable_failure_policy")
        assert not found
        assert policy_raw == ""
        await store.close()


def test_consume_task_runtime_hints_surfaces_replan_and_routing_once():
    asyncio.run(_consume_task_runtime_hints_surfaces_replan_and_routing_once())


async def _consume_task_runtime_hints_surfaces_replan_and_routing_once():
    from core.loop.task.runtime import (
        _consume_task_runtime_hints,
        _ingest_actionable_meta_reflections,
    )
    from memory.working import WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("重规划任务", goal="先复盘再继续", next_step="旧步骤")
        await store.add_meta_reflection(
            reflection_id="mr-routing-tasksplit",
            target_kind="task_split",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="任务拆分不完整",
            proposal="先定位资源，再读取文件",
            verification_plan="确认不再出现 EmptyPath",
            decision="apply",
            task_id=task_id,
            run_id=1,
            tool_name="file.read",
        )
        await store.add_meta_reflection(
            reflection_id="mr-routing-guard",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="动作选择漂移",
            proposal="切换到 repair tier 复核动作",
            verification_plan="确认 chosen_action_id 合法",
            decision="apply",
            task_id=task_id,
            run_id=2,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        # list_meta_reflections 按 (created_at, id) ASC 排序，id 是 string；
        # 字母序 guard < tasksplit，顺序不固定，用 set 比较
        assert set(injected) == {"mr-routing-tasksplit", "mr-routing-guard"}

        task = await store.get_task_by_id(task_id)
        task = await _consume_task_runtime_hints(store, task, wm)
        assert task is not None
        assert task.next_step == "旧步骤"
        assert task.model_tier == "repair"
        assert task.extras["last_replan_reflection_id"] == "mr-routing-tasksplit"
        assert task.extras["last_routing_reflection_id"] == "mr-routing-guard"

        top = wm.get_top()
        assert any(item["kind"] == "task_replan" for item in top)
        assert any(item["kind"] == "routing_guard" for item in top)
        assert any(item["kind"] == "task_replan" and "task.update" in item["content"] for item in top)
        assert any(item["kind"] == "routing_guard" and "已自动写回 task.model_tier=repair" in item["content"] for item in top)

        again = await _consume_task_runtime_hints(store, task, wm)
        assert again is not None
        assert again.next_step == "旧步骤"
        assert again.model_tier == "repair"
        await store.close()


def test_meta_reflection_threshold_apply_queues_explicit_policy_hint():
    asyncio.run(_meta_reflection_threshold_apply_queues_explicit_policy_hint())


async def _meta_reflection_threshold_apply_queues_explicit_policy_hint():
    from core.loop.task.runtime import _ingest_actionable_meta_reflections
    from memory.working import WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-threshold-explicit",
            target_kind="threshold",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="静默窗口过早触发",
            proposal="确认外部状态是否恢复；若频繁误杀，则调整 durable failure 阈值或静默策略。",
            verification_plan="等待静默窗口结束后重跟，并比较是否仍被直接跳过。",
            decision="apply",
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-threshold-explicit"]

        top = wm.get_top()
        # 阈值策略采用增量方式：默认 threshold=3 一次尝试后变为 4
        assert any(
            item["kind"] == "meta_reflection"
            and "threshold=4" in item["content"]
            and "memory.set_fact" in item["content"]
            for item in top
        )

        raw, found = await store.get_fact("control:meta_reflection_hint:threshold")
        assert found
        # 增量输出： threshold=3+1=4, ttl_sec=7200//2=3600
        assert json.loads(raw)["suggested_policy"] == {"threshold": 4, "ttl_sec": 3600}

        policy_raw, found = await store.get_fact("control:durable_failure_policy")
        assert not found
        assert policy_raw == ""
        await store.close()


def test_consume_task_runtime_hints_surfaces_preferred_tier_hint():
    asyncio.run(_consume_task_runtime_hints_surfaces_preferred_tier_hint())


async def _consume_task_runtime_hints_surfaces_preferred_tier_hint():
    from core.loop.task.runtime import (
        _consume_task_runtime_hints,
        _ingest_actionable_meta_reflections,
    )
    from memory.working import WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("路由护栏任务", goal="切换 tier")
        await store.add_meta_reflection(
            reflection_id="mr-routing-tier",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="reader 过轻导致动作漂移",
            proposal="切换到 reasoner tier 复核动作",
            verification_plan="确认动作选择回到合法工具集",
            decision="apply",
            task_id=task_id,
            run_id=3,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-routing-tier"]

        task = await store.get_task_by_id(task_id)
        task = await _consume_task_runtime_hints(store, task, wm)
        assert task is not None
        assert task.model_tier == "reasoner"
        top = wm.get_top()
        assert any(item["kind"] == "routing_guard" and "reasoner" in item["content"] for item in top)
        assert any(item["kind"] == "routing_guard" and "已自动写回 task.model_tier=reasoner" in item["content"] for item in top)
        await store.close()


def test_consume_task_runtime_hints_surfaces_task_meta_reflection_to_wm():
    asyncio.run(_consume_task_runtime_hints_surfaces_task_meta_reflection_to_wm())


async def _consume_task_runtime_hints_surfaces_task_meta_reflection_to_wm():
    from core.loop.task.runtime import (
        _consume_task_runtime_hints,
        _ingest_actionable_meta_reflections,
    )
    from memory.working import WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        task_id = await store.add_task("反思显化任务", goal="让 LLM 感知 task 级反思")
        await store.add_meta_reflection(
            reflection_id="mr-task-visible",
            target_kind="routing",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="动作选择漂移",
            proposal="切换到 repair tier 复核动作",
            verification_plan="确认 chosen_action_id 合法",
            decision="apply",
            task_id=task_id,
            run_id=8,
            tool_name="file.read",
        )

        wm = WorkingMemory(capacity=10)
        injected = await _ingest_actionable_meta_reflections(store, wm)
        assert injected == ["mr-task-visible"]

        task = await store.get_task_by_id(task_id)
        task = await _consume_task_runtime_hints(store, task, wm)
        assert task is not None
        assert task.extras["last_task_meta_reflection_id"] == "mr-task-visible"
        top = wm.get_top()
        assert any(item["kind"] == "task_reflection" and "mr-task-visible" not in item["content"] for item in top)
        await store.close()


def test_ingest_actionable_meta_reflections_queues_generic_control_hint():
    asyncio.run(_ingest_actionable_meta_reflections_queues_generic_control_hint())


async def _ingest_actionable_meta_reflections_queues_generic_control_hint():
    from core.loop.task.runtime import _ingest_actionable_meta_reflections
    from memory.working import WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        await store.add_meta_reflection(
            reflection_id="mr-prompt-hint",
            target_kind="prompt",
            trigger="failure_pattern",
            loop_level="double",
            diagnosis="提示词规则需要修订",
            proposal="收紧 JSON schema 说明",
            verification_plan="确认 parse failure 下降",
            decision="apply",
            tool_name="judgment",
        )

        injected = await _ingest_actionable_meta_reflections(store, WorkingMemory(capacity=10))
        assert injected == ["mr-prompt-hint"]

        raw, found = await store.get_fact("control:meta_reflection_hint:prompt")
        assert found
        payload = json.loads(raw)
        assert payload["reflection_id"] == "mr-prompt-hint"
        assert payload["proposal"] == "收紧 JSON schema 说明"
        await store.close()


def test_execution_background_run_does_not_record_completion_early():
    asyncio.run(_execution_background_run_does_not_record_completion_early())


async def _execution_background_run_does_not_record_completion_early():
    import os

    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.exec import ProcessManager, process_kill
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        os.environ["LINGZHOU_PROCESS_STATE_DIR"] = str(root / "proc-state")
        ProcessManager.clear()
        store = TaskStore(root / "runtime.db")
        episodic = EpisodicMemory(root)
        semantic = cast("Any", SemanticMemory(root))
        await store.open()
        await store.add_task("后台执行", goal="启动后保持 running")
        reg = ToolRegistry()
        reg.discover(_proj_root() / "tools")
        layer = _execution_layer(reg)
        ctx = _tool_ctx(debug=False, task_store=store, episodic=episodic, semantic=semantic)
        action = _judgment_output(
            decision="act",
            chosen_action_id="exec",
            params={"command": 'python3 -c "import time; time.sleep(5)"', "background": True, "timeout": 5},
            rationale="launch background exec",
        )

        result = await layer.dispatch(action, ctx)
        assert result.error is None

        run_id = int(result.metadata["run_id"])
        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"

        completed = episodic.list_events("run_completed", limit=5)
        failed = episodic.list_events("run_failed", limit=5)
        assert completed == []
        assert failed == []

        node = semantic.get(f"run-result-{run_id}")
        assert node is None

        await process_kill({"process_id": str(result.metadata["process_id"] or "")}, _tool_ctx())
        await store.close()


def test_exec_process_write_pty_roundtrip():
    import subprocess
    script = r'''
import asyncio, json
from types import SimpleNamespace
from tools.exec import exec_run, process_write, process_poll, process_log, _MANAGER
from tools.registry import ToolContext

async def main():
    _MANAGER.clear()
    ctx = ToolContext(
        config=SimpleNamespace(loop=SimpleNamespace(act=True)),
        wm=None, task_store=None, episodic=None, semantic=None, emotion=None,
    )
    res = await exec_run({
        "command": "bash -lc 'stty -echo; read a; echo got:$a'",
        "background": True,
        "pty": True,
        "timeout": 5,
    }, ctx)
    sid = json.loads(res.evidence)["process_id"]
    await asyncio.sleep(0.2)
    await process_write({"process_id": sid, "data": "hi\\n"}, ctx)
    for _ in range(60):
        poll = await process_poll({"process_id": sid}, ctx)
        status = json.loads(poll.summary)
        if status["status"] == "finished":
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.1)
    log = await process_log({"process_id": sid, "offset": 0, "limit": 400}, ctx)
    print(log.summary)

asyncio.run(main())
'''
    out = subprocess.check_output(["python3", "-c", script], cwd=str(_proj_root()), text=True)
    assert "TimeoutError" not in out
    assert "ProcessNotFound" not in out


def test_skill_registry():
    from core.skill import SkillRegistry
    reg = SkillRegistry()
    # 冷启动场景
    skills = reg.match_for_context(wm_pressure=0.05, has_active_task=False,
                                    has_next_step=False, failure_count=0, high_error_streak=0,
                                    max_inject=3)
    assert any(s.name == "runtime-bootstrap" for s in skills)
    # 失败场景
    skills_fail = reg.match_for_context(wm_pressure=0.5, has_active_task=True,
                                         has_next_step=True, failure_count=3, high_error_streak=3,
                                         max_inject=3)
    assert any(s.name == "failure-reflection" for s in skills_fail)


def test_skill_registry_does_not_stick_failure_reflection_without_failures():
    from core.skill import SkillRegistry

    reg = SkillRegistry()
    skills = reg.match_for_context(
        last_applied=["failure-reflection"],
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=True,
        failure_count=0,
        high_error_streak=0,
        context_text="继续推进当前任务，执行下一步并保持状态连续。",
        max_inject=2,
    )
    names = [skill.name for skill in skills]
    assert "task-continuity" in names
    assert "failure-reflection" not in names


def test_skill_registry_loads_package_skill_and_matches_context(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"

    pkg = skills_dir / "karpathy-coding-base"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: karpathy-coding-base
description: |
  Andrej Karpathy-inspired coding guardrails. Use when: fixing bugs, refactoring, writing scripts, code review.
---
先思考，再编码。极简优先。手术式变更。目标驱动验证。
""",
        encoding="utf-8",
    )

    pkg2 = skills_dir / "interaction"
    pkg2.mkdir(parents=True)
    (pkg2 / "SKILL.md").write_text(
        """---
name: interaction
description: |
  统一人际交互入口。Use when: 提问/确认/好奇追问/理解语境。
---
先判断意图，再决定是回答、提问还是确认。方向不清时问一个最小问题。
""",
        encoding="utf-8",
    )

    pkg3 = skills_dir / "proactive-work"
    pkg3.mkdir(parents=True)
    (pkg3 / "SKILL.md").write_text(
        """---
name: proactive-work
description: |
  主动工作方法论。Use when: 完成任务后、等回复时、需自主决定下一步。
---
完成任务后不要等待，主动判断并推进下一步。
""",
        encoding="utf-8",
    )

    pkg4 = skills_dir / "self-monitoring"
    pkg4.mkdir(parents=True)
    (pkg4 / "SKILL.md").write_text(
        """---
name: self-monitoring
description: |
  Self-monitoring. Use when: tool call fails, edit fails, log errors, execution deviation.
state_rules: failure_signal >= 0.5 => 1.0
---
发现漂移、日志错误、编辑失败后，先检查并修复。
""",
        encoding="utf-8",
    )

    pkg5 = skills_dir / "error-handling"
    pkg5.mkdir(parents=True)
    (pkg5 / "SKILL.md").write_text(
        """---
name: error-handling
description: |
  Error handling. Use when: exec denied, timeout, permission error, tool call fails.
state_rules: failure_signal >= 0.5 => 2.0
---
失败后先分类错误，再决定重试、替代还是汇报。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    # description 关键词只进入 catalog，不驱动机器侧选择。
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="请你修复 bug，并顺手重构这个脚本",
        max_inject=0,
    )
    assert skills == []

    interaction_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="我有点好奇，你觉得这里真正的分歧是什么？",
        max_inject=0,
    )
    assert interaction_skills == []

    proactive_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="做完了当前任务，接下来你自己判断往前推进",
        max_inject=0,
    )
    assert proactive_skills == []

    monitor_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="这次 edit 失败了，日志也有异常，帮我看看哪里偏了",
        max_inject=20,
    )
    assert any(s.name == "self-monitoring" for s in monitor_skills)

    err_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="exec 被拒绝了，还报了 timeout 和 permission error",
        max_inject=20,
    )
    assert any(s.name == "error-handling" for s in err_skills)

    # error-handling 有更高的 state_rules 权重（2.0 > 1.0），在 failure 信号下应排第一
    focused_err_skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=1,
        high_error_streak=1,
        context_text="exec 被拒绝了，还报了 timeout 和 permission error",
        max_inject=1,
    )
    assert [skill.name for skill in focused_err_skills] == ["error-handling"]


def test_skill_registry_parses_state_bias_from_frontmatter(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "continuity-driven"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: continuity-driven
description: |
  Declarative state bias for next-step continuity.
state_bias: has_active_task=1.0, has_next_step=4.5
---
当任务有明确 next_step 时，先推进当前任务，不要切走。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=True,
        failure_count=0,
        high_error_streak=0,
        context_text="",
        max_inject=1,
    )

    assert [skill.name for skill in skills] == ["continuity-driven"]


def test_skill_registry_state_rules_support_thresholds_and_inhibition(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "failure-gated"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: failure-gated
description: Focus on gateway reconnect storms after enough failure evidence appears.
state_rules: |
  failure_signal_ratio >= 0.6 => 1.8
  inhibit if has_next_step >= 1 and failure_signal_ratio < 0.6
---
先在失败证据充分时再接管。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)

    blocked = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=True,
        failure_count=0,
        high_error_streak=0,
        context_text="gateway reconnect storm websocket flapping",
        max_inject=1,
    )
    assert blocked == []

    allowed = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=True,
        failure_count=3,
        high_error_streak=2,
        context_text="gateway reconnect storm websocket flapping",
        max_inject=1,
    )
    assert [skill.name for skill in allowed] == ["failure-gated"]


def test_skill_registry_state_rules_support_negative_weights(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "idle-helper"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: idle-helper
description: Only make sense when no active task is present.
state_rules: |
  has_active_task => -2.0
---
活跃任务存在时主动降权。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="",
        max_inject=1,
    )

    assert skills == []


def test_skill_registry_loads_seed_skill_from_file():
    from core.skill import SkillRegistry

    reg = SkillRegistry()
    skill = next(skill for skill in reg.all_skills() if skill.name == "runtime-bootstrap")

    assert skill.origin == "seed"
    assert skill.source_path.endswith("runtime-bootstrap/SKILL.md")
    assert skill.guidance == ""
    assert "bootstrap_identity" in skill.load_guidance()


def test_builtin_skills_follow_unified_frontmatter_spec():
    from core.skill import SkillRegistry, _iter_skill_files, _split_frontmatter

    skills_dir = _proj_root() / "prompts" / "skills"
    required = (
        "name",
        "description",
        "compatibility",
        "tags",
        "triggers",
        "state_rules",
    )

    expected_names: set[str] = set()
    for path in _iter_skill_files(skills_dir):
        meta, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        missing = [key for key in required if not str(meta.get(key, "")).strip()]
        assert not missing, f"{path} missing frontmatter keys: {missing}"
        assert "Use when" in str(meta["description"]), f"{path} description must contain 'Use when'"
        expected_names.add(str(meta["name"]).strip())

    reg = SkillRegistry(skills_dir=skills_dir)
    loaded_names = {skill.name for skill in reg.all_skills()}

    assert loaded_names == expected_names


def test_skill_registry_activate_loads_guidance_and_resources(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "sample-skill"
    (pkg / "references").mkdir(parents=True)
    (pkg / "references" / "REFERENCE.md").write_text("skill reference", encoding="utf-8")
    (pkg / "SKILL.md").write_text(
        """---
name: sample-skill
description: Use when the task needs sample skill activation.
compatibility: Requires Lingzhou with skill.activate support.
allowed-tools: file.read skill.activate
metadata:
  author: example-org
  version: "1.0"
---
先阅读 REFERENCE.md，再决定如何执行。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skill = reg.get("sample-skill")

    assert skill is not None
    assert skill.metadata == {"author": "example-org", "version": "1.0"}
    assert skill.allowed_tools == ["file.read", "skill.activate"]

    activated, text = reg.activate("sample-skill")
    assert activated is not None
    assert "<skill_content name=\"sample-skill\">" in text
    assert "先阅读 REFERENCE.md" in text
    assert "references/REFERENCE.md" in text


def test_workspace_skill_can_override_builtin_definition(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "runtime-bootstrap"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: runtime-bootstrap
description: 覆盖版 bootstrap skill
tags: custom
triggers: 冷启动
state_bias: idle_only=0.2
---
先审视工作记忆中的身份，再自行决定是否立刻创建任务。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skill = next(skill for skill in reg.all_skills() if skill.name == "runtime-bootstrap")

    assert skill.origin == "workspace"
    assert skill.description == "覆盖版 bootstrap skill"


def test_skill_registry_does_not_backfill_seed_skills_when_workspace_present(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "solo-skill"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: solo-skill
description: 只有 workspace skill，不应自动回填 seed。
triggers: 单独运行
---
只根据 workspace 当前 skill 集合工作。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    names = [skill.name for skill in reg.all_skills()]

    assert names == ["solo-skill"]


def test_seed_workspace_skills_only_on_empty_workspace(tmp_path):
    from core.skill import seed_workspace_skills

    written = seed_workspace_skills(tmp_path)
    skills_dir = tmp_path / "skills"
    assert written >= 5
    assert (skills_dir / "runtime-bootstrap" / "SKILL.md").exists()

    custom_dir = skills_dir / "custom"
    custom_dir.mkdir(parents=True)
    (custom_dir / "SKILL.md").write_text(
        """---
name: custom
description: keep workspace authoritative
---
workspace authoritative
""",
        encoding="utf-8",
    )
    written_again = seed_workspace_skills(tmp_path)
    assert written_again == 0


def test_seed_workspace_skills_updates_unmodified_workspace_copy(monkeypatch, tmp_path):
    import core.skill as skill_mod

    seed_dir = tmp_path / "seed"
    seed_skill_dir = seed_dir / "runtime-bootstrap"
    seed_skill_dir.mkdir(parents=True)
    seed_file = seed_skill_dir / "SKILL.md"
    seed_file.write_text(
        """---
name: runtime-bootstrap
description: seed v1
---
seed v1
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(skill_mod, "_seed_skills_dir", lambda: seed_dir)

    written = skill_mod.seed_workspace_skills(tmp_path)
    target = tmp_path / "skills" / "runtime-bootstrap" / "SKILL.md"
    assert written == 1
    assert "seed v1" in target.read_text(encoding="utf-8")

    seed_file.write_text(
        """---
name: runtime-bootstrap
description: seed v2
---
seed v2
""",
        encoding="utf-8",
    )

    written_again = skill_mod.seed_workspace_skills(tmp_path)
    assert written_again == 1
    assert "seed v2" in target.read_text(encoding="utf-8")


def test_seed_workspace_skills_preserves_workspace_override_on_seed_change(monkeypatch, tmp_path):
    import core.skill as skill_mod

    seed_dir = tmp_path / "seed"
    seed_skill_dir = seed_dir / "runtime-bootstrap"
    seed_skill_dir.mkdir(parents=True)
    seed_file = seed_skill_dir / "SKILL.md"
    seed_file.write_text(
        """---
name: runtime-bootstrap
description: seed v1
---
seed v1
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(skill_mod, "_seed_skills_dir", lambda: seed_dir)

    skill_mod.seed_workspace_skills(tmp_path)
    target = tmp_path / "skills" / "runtime-bootstrap" / "SKILL.md"
    target.write_text(
        """---
name: runtime-bootstrap
description: workspace override
---
workspace override
""",
        encoding="utf-8",
    )

    seed_file.write_text(
        """---
name: runtime-bootstrap
description: seed v2
---
seed v2
""",
        encoding="utf-8",
    )

    written_again = skill_mod.seed_workspace_skills(tmp_path)
    assert written_again == 0
    assert "workspace override" in target.read_text(encoding="utf-8")


def test_skill_registry_description_triggers_no_longer_drive_machine_selection(tmp_path):
    """LLM 新范式：description 中的 Triggers: 文本不再驱动机器侧评分。
    skill 无 state_rules 时 score=0，max_inject=1 下不会被选中。
    激活改由 LLM 阅读 catalog description 后自主调用 skill.activate 完成。
    """
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "error-handling"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: error-handling
description: |
  Error handling. Triggers: exec denied, timeout, permission error
---
失败后先分类错误，再决定重试、替代还是汇报。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=True,
        failure_count=0,
        high_error_streak=0,
        context_text="exec 被拒绝了，还报了 timeout 和 permission error",
        max_inject=1,
    )

    # 新范式：description Triggers: 文本不再驱动机器侧选择；无 state_rules = score 0 = 不被注入
    assert [skill.name for skill in skills] != ["error-handling"]


def test_skill_registry_does_not_match_on_skill_name_alone(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "error-handling"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: error-handling
description: 完全中性的技能说明。
---
这里没有声明任何 triggers、tags 或 match_terms。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="error handling timeout permission error",
        max_inject=1,
    )

    assert [skill.name for skill in skills] != ["error-handling"]


def test_skill_registry_match_terms_no_longer_drive_machine_selection(tmp_path):
    """LLM 新范式：match_terms 不再驱动机器侧评分。
    skill 无 state_rules 时 score=0，max_inject=1 不会被选中，可以安全删除 match_terms 字段。
    """
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "neutral-skill"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: neutral-skill
description: 中性说明，不依赖 skill 名称词面命中。
match_terms: timeout, permission error, exec denied
---
根据显式 match_terms 感知上下文。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="exec 被拒绝了，还报了 timeout 和 permission error",
        max_inject=1,
    )

    # match_terms 不再驱动机器侧评分；无 state_rules = score 0 = 不会被选中
    assert [skill.name for skill in skills] != ["neutral-skill"]


def test_skill_registry_match_rules_no_longer_drive_machine_selection(tmp_path):
    """LLM 新范式：match_rules 不再驱动机器侧评分。
    skill 无 state_rules 时 score=0，max_inject=1 不会被选中，可以安全删除 match_rules 字段。
    """
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "gateway-reconnect"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: gateway-reconnect
description: Diagnose gateway reconnect storms in worker processes.
match_rules: |
    all: gateway | reconnect | websocket | flapping => 2.2
---
聚焦网关重连风暴，依赖显式 match_rules 而不是 description 自动拆词。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="gateway worker reconnect storm websocket flapping",
        max_inject=1,
    )

    # match_rules 不再驱动机器侧评分；无 state_rules = score 0 = 不会被选中
    assert [skill.name for skill in skills] != ["gateway-reconnect"]


def test_skill_registry_description_only_no_longer_matches_context(tmp_path):
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "gateway-reconnect"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: gateway-reconnect
description: Diagnose gateway reconnect storms and websocket flapping in worker processes.
---
如果没有显式 match_rules / match_terms / triggers，就不应靠 description 自动命中。
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    skills = reg.match_for_context(
        wm_pressure=0.1,
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        context_text="gateway worker reconnect storm websocket flapping",
        max_inject=1,
    )

    assert [skill.name for skill in skills] != ["gateway-reconnect"]


# ══════════════════════════════════════════════════════════════════════════════
# skill 触发优化回归：last_applied 语义、catalog 排序、primary_skill LLM 记忆
# ══════════════════════════════════════════════════════════════════════════════

def test_last_applied_boost_requires_existing_score(tmp_path):
    """last_applied 只能加权已有得分的 skill，不能凭空浮出零分 skill。"""
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    # skill-A：有 state_rules 会自然得分
    pkg_a = skills_dir / "skill-a"
    pkg_a.mkdir(parents=True)
    (pkg_a / "SKILL.md").write_text(
        """---
name: skill-a
description: |
  Handles active tasks. Use when: task is running.
compatibility: any
tags: task
triggers: ["active task"]
match_terms: ""
match_rules: ""
state_rules: |
    has_active_task => 0.5
---
A body.
""",
        encoding="utf-8",
    )
    # skill-b：无任何匹配规则，完全零分
    pkg_b = skills_dir / "skill-b"
    pkg_b.mkdir(parents=True)
    (pkg_b / "SKILL.md").write_text(
        """---
name: skill-b
description: |
  Ultra-niche skill. Use when: very rare.
compatibility: any
tags: rare
triggers: ["rare event"]
match_terms: ""
match_rules: ""
state_rules: ""
---
B body.
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)

    # 场景：has_active_task=True，last_applied 包含 skill-b（上轮 LLM 碰巧 activate 过）
    # skill-b 无任何信号→ 即使在 last_applied 中也不能浮出
    skills = reg.match_for_context(
        last_applied=["skill-b"],
        has_active_task=True,
        has_next_step=False,
        failure_count=0,
        wm_pressure=0.1,
        context_text="",
        max_inject=2,
    )
    names = [s.name for s in skills]
    assert "skill-a" in names, "skill-a 应因 state 信号得分"
    assert "skill-b" not in names, "skill-b 无信号，last_applied 不能独立浮出它"


def test_last_applied_boosts_skill_when_it_already_has_score(tmp_path):
    """last_applied 加权逻辑：skill 有正分时，last_applied 应让它排名更靠前。"""
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    # skill-weak：idle 状态下有基础分
    pkg_w = skills_dir / "skill-weak"
    pkg_w.mkdir(parents=True)
    (pkg_w / "SKILL.md").write_text(
        """---
name: skill-weak
description: |
  Weak skill. Use when: idle, no task active.
compatibility: any
tags: debug
triggers: ["idle"]
state_rules: idle_only => 0.5
---
Weak body.
""",
        encoding="utf-8",
    )
    # skill-strong：同样有 idle 基础分，且在 last_applied 中（加权后排第一）
    pkg_s = skills_dir / "skill-strong"
    pkg_s.mkdir(parents=True)
    (pkg_s / "SKILL.md").write_text(
        """---
name: skill-strong
description: |
  Strong skill. Use when: idle, no active task.
compatibility: any
tags: bug
triggers: ["idle"]
state_rules: idle_only => 0.5
---
Strong body.
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)

    # has_active_task=False, has_next_step=False → idle_only=1.0 → 两个 skill 都有基础分
    skills = reg.match_for_context(
        last_applied=["skill-strong"],
        has_active_task=False,
        has_next_step=False,
        failure_count=0,
        wm_pressure=0.1,
        context_text="debug bug fix",
        max_inject=2,
    )
    names = [s.name for s in skills]
    assert names[0] == "skill-strong", "last_applied 加权后应排第一"
    assert "skill-weak" in names


def test_skill_catalog_pinned_mark_appears_for_last_applied():
    """catalog 格式中，last_applied 的 skill 应带 [↑] 标记。"""
    from core.judgment.context.skills import _fmt_skill_catalog
    from core.skill import SkillRegistry

    # 构建两个最小 skill
    reg = SkillRegistry()
    all_skills = reg.all_skills()
    if len(all_skills) < 2:
        pytest.skip("需要至少 2 个内置 skill")

    skill_a, skill_b = all_skills[0], all_skills[1]
    catalog_with_pin = _fmt_skill_catalog([skill_a, skill_b], pinned_names={skill_a.name})
    catalog_no_pin = _fmt_skill_catalog([skill_a, skill_b], pinned_names=None)

    assert "`[↑]`" in catalog_with_pin, "pinned skill 应有 [↑] 标记"
    assert skill_a.name in catalog_with_pin
    # skill_b 没有 pin，不应该有 [↑]
    # 找到 skill_b 那行，确认没有 [↑]
    b_line = next(line_text for line_text in catalog_with_pin.splitlines() if skill_b.name in line_text)
    assert "`[↑]`" not in b_line, "未 pinned 的 skill 不应有 [↑] 标记"
    assert "`[↑]`" not in catalog_no_pin, "无 pinned_names 时不应有 [↑] 标记"


def test_skill_catalog_sorted_by_match_score(tmp_path):
    """match_for_context 返回的 state-scored skills 排在 catalog 顶部，其余按原序追加。"""
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    for name, has_state in [("skill-x", False), ("skill-y", False), ("skill-z", True)]:
        state_rules = "wm_pressure_ratio >= 0.1 => 1.0" if has_state else ""
        pkg = skills_dir / name
        pkg.mkdir(parents=True)
        (pkg / "SKILL.md").write_text(
            f"""---
name: {name}
description: |
  {name} handler. Use when: relevant state signal fires.
compatibility: any
tags: test
triggers: ["signal"]
state_rules: {state_rules}
---
body.
""",
            encoding="utf-8",
        )

    reg = SkillRegistry(skills_dir=skills_dir)

    # wm_pressure=0.1 → wm_pressure_ratio=0.25 >= 0.1 → skill-z 得分 > 0，排第一
    matched = reg.match_for_context(
        context_text="",
        has_active_task=False,
        has_next_step=False,
        failure_count=0,
        wm_pressure=0.1,
        max_inject=3,
    )
    names = [s.name for s in matched]
    assert names == ["skill-z"], "只返回 state 信号命中的 skill"


def test_primary_skill_uses_last_applied_memory(tmp_path):
    """primary_skill_section 应基于 LLM 上轮记忆（last_applied），而不是 keyword 预选。"""
    from core.judgment.context.skills import _fmt_primary_skill
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    for name, term in [("alpha-skill", "alpha"), ("beta-skill", "beta")]:
        pkg = skills_dir / name
        pkg.mkdir(parents=True)
        (pkg / "SKILL.md").write_text(
            f"""---
name: {name}
description: |
  {name} processor. Use when: dealing with {term} scenarios.
compatibility: any
tags: test
triggers: ["{term}"]
match_terms: {term}
match_rules: ""
state_rules: ""
---
body.
""",
            encoding="utf-8",
        )

    reg = SkillRegistry(skills_dir=skills_dir)

    # 模拟 last_applied = ["beta-skill"]，但当前 context 只包含 alpha 关键词
    # primary_skill 应是 beta-skill（LLM 记忆优先），不是 alpha-skill（keyword 命中）
    last_applied = ["beta-skill"]
    primary_skill = reg.get(last_applied[0]) if last_applied else None
    text = _fmt_primary_skill(primary_skill)

    assert "beta-skill" in text, "primary_skill 应展示 LLM 上轮选择的 beta-skill"
    assert "skill.activate" in text, "应包含 activation 提示"

    # 当 last_applied 为空时，primary_skill 为 None → 返回兜底文本
    empty_primary = _fmt_primary_skill(None)
    assert "无明显 skill 候选" in empty_primary


def test_primary_skill_none_when_no_last_applied():
    """无 last_applied 历史时，primary_skill 降级为空文本提示，不崩溃。"""
    from core.judgment.context.skills import _fmt_primary_skill

    result = _fmt_primary_skill(None)
    assert result  # 有内容
    assert "skill.activate" not in result or "无明显" in result


def test_match_for_context_catalog_order_vs_original(tmp_path):
    """catalog 排序：state-scored skill 置顶，zero-score skill 跟随，且总数不变。"""
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    names_terms = [("aa", False), ("bb", False), ("cc", True), ("dd", False)]
    for name, has_state in names_terms:
        state_rules = "wm_pressure_ratio >= 0.1 => 1.0" if has_state else ""
        pkg = skills_dir / name
        pkg.mkdir(parents=True)
        (pkg / "SKILL.md").write_text(
            f"""---
name: {name}
description: |
  {name} handler. Use when: state signal fires.
compatibility: any
tags: test
triggers: ["signal"]
state_rules: {state_rules}
---
body.
""",
            encoding="utf-8",
        )

    reg = SkillRegistry(skills_dir=skills_dir)
    # wm_pressure=0.1 → wm_pressure_ratio=0.25 ≥ 0.1 → cc 得分 > 0
    result = reg.match_for_context(
        context_text="",
        has_active_task=False,
        has_next_step=False,
        failure_count=0,
        wm_pressure=0.1,
        max_inject=3,
    )
    assert [skill.name for skill in result] == ["cc"], "只返回 state 信号命中的 skill"


def test_skill_registry_does_not_stick_any_skill_without_signal(tmp_path):
    """无 state 无 context 无 last_applied 匹配时，max_inject>0 返回空列表。"""
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "niche-skill"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: niche-skill
description: |
  Very niche. Use when: extremely rare circumstance.
compatibility: any
tags: rare
triggers: ["ultra rare"]
match_terms: ultrararekeyword
match_rules: ""
state_rules: ""
---
body.
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)

    # 完全冷、无 context、无 last_applied
    skills = reg.match_for_context(
        last_applied=[],
        has_active_task=False,
        has_next_step=False,
        failure_count=0,
        wm_pressure=0.05,
        context_text="everyday normal task",
        max_inject=2,
    )
    assert not skills, "无任何匹配信号时 max_inject>0 应返回空（不强行注入）"


def test_builtin_skill_catalog_coverage():
    """所有内置 skill 均应出现在显式 catalog 中。"""
    from core.skill import SkillRegistry

    reg = SkillRegistry()
    all_names = {s.name for s in reg.all_skills()}
    assert all_names
    assert all_names == {s.name for s in reg.all_skills()}


def test_skill_catalog_section_contains_activation_hint():
    """catalog 格式字符串应包含 skill.activate 提示，告知 LLM 主动激活。"""
    from core.judgment.context.skills import _fmt_skill_catalog
    from core.skill import SkillRegistry

    reg = SkillRegistry()
    catalog = _fmt_skill_catalog(reg.all_skills(), pinned_names=None)

    assert "skill.activate" in catalog
    assert "AGENT SKILLS CATALOG" in catalog


def test_failure_reflection_sticks_with_failures():
    """有失败信号时，failure-reflection 应正常命中，不被误过滤。"""
    from core.skill import SkillRegistry

    reg = SkillRegistry()
    skills = reg.match_for_context(
        last_applied=[],
        has_active_task=True,
        has_next_step=True,
        failure_count=4,
        high_error_streak=4,
        wm_pressure=0.5,
        context_text="",
        max_inject=3,
    )
    names = [s.name for s in skills]
    assert "failure-reflection" in names, "有失败信号时 failure-reflection 应被选中"


def test_runtime_bootstrap_selected_when_idle():
    """冷启动（无任务、无失败）时 runtime-bootstrap 应被选中。"""
    from core.skill import SkillRegistry

    reg = SkillRegistry()
    skills = reg.match_for_context(
        context_text="",
        wm_pressure=0.05,
        has_active_task=False,
        has_next_step=False,
        failure_count=0,
        high_error_streak=0,
        max_inject=3,
    )
    assert any(s.name == "runtime-bootstrap" for s in skills)


def test_last_applied_not_in_result_if_no_signal_and_max_inject_set(tmp_path):
    """max_inject>0 时，last_applied 里的 skill 若无信号得分，不能进入结果。"""
    from core.skill import SkillRegistry

    skills_dir = tmp_path / "skills"
    pkg = skills_dir / "ghost-skill"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        """---
name: ghost-skill
description: |
  Ghost skill. Use when: haunted codebase.
compatibility: any
tags: ghost
triggers: ["haunt"]
match_terms: haunt
match_rules: ""
state_rules: ""
---
ghost body.
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(skills_dir=skills_dir)
    # ghost-skill 上轮 LLM 用过，但本轮没有任何匹配信号
    result = reg.match_for_context(
        last_applied=["ghost-skill"],
        has_active_task=False,
        has_next_step=False,
        failure_count=0,
        wm_pressure=0.05,
        context_text="totally unrelated context here",
        max_inject=2,
    )
    names = [s.name for s in result]
    assert "ghost-skill" not in names, "无信号时 last_applied 不能强行注入（score>0 guard）"
