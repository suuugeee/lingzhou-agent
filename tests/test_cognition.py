"""认知循环、chat reply、resolve 等集成测试"""
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

def test_bootstrap_wm_injection():
    from memory.working import WorkingMemory, WMItem

    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "BOOTSTRAP.md").write_text("# Bootstrap\n你是灵舟。", encoding="utf-8")
        (ws / "SOUL.md").write_text("# Soul\n真实 0.85", encoding="utf-8")

        wm = WorkingMemory(capacity=20)
        for fname in ("BOOTSTRAP.md", "IDENTITY.md", "SOUL.md"):
            fpath = ws / fname
            if fpath.exists():
                content = fpath.read_text(encoding="utf-8")
                # 每个文件用不同 kind，避免 WorkingMemory 按 kind 去重
                wm.add(WMItem(kind=f"bootstrap_identity:{fname}",
                               content=f"[{fname}]\n{content[:400]}", priority=1.0))

        items = wm.get_top(10)
        assert sum(1 for i in items if i["kind"].startswith("bootstrap_identity")) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 完整构造链路（不调 LLM）
# ══════════════════════════════════════════════════════════════════════════════

def test_cognition_loop_init():
    """CognitionLoop.__init__ 不崩溃，关键参数正确传递。"""
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        assert loop.semantic.decay_lambda == cfg.memory.semantic_decay_lambda
        assert loop.episodic.max_events == cfg.memory.max_events


def test_curiosity_signal_does_not_auto_create_task():
    asyncio.run(_curiosity_signal_does_not_auto_create_task())


async def _curiosity_signal_does_not_auto_create_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            loop._idle_cycles = cfg.thresholds.curiosity_idle_min_cycles
            loop._last_curiosity_signal_idle_cycle = 0
            ethos_state = cast(Any, SimpleNamespace(
                values=SimpleNamespace(curiosity=cfg.thresholds.curiosity_idle_task + 0.1)
            ))

            await loop._maybe_curiosity_task(ethos_state)

            tasks = await loop.task_store.list_tasks(limit=20)
            assert tasks == []
            # WM 注入已移除（016e0a56），改为纯事实观测；
            # 验证 _last_curiosity_signal_idle_cycle 已被标记（防止重复触发）
            assert loop._last_curiosity_signal_idle_cycle == loop._idle_cycles
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_dev_model_switch_syncs_routing_entries_following_primary_model():
    from cli.dev import _sync_routing_models_on_primary_switch

    cfg_data = {
        "model": "copilot/gpt-5.4",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
            "repair": "copilot/gpt-5.4",
        },
    }

    changed = _sync_routing_models_on_primary_switch(
        cfg_data,
        old_model="copilot/gpt-5.4",
        new_model="copilot/gpt-5.4-mini",
    )

    assert changed == ["reasoner", "repair"]
    assert cfg_data["routing"]["reader"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/gpt-5.4-mini"
    assert cfg_data["routing"]["repair"] == "copilot/gpt-5.4-mini"


def test_dev_model_switch_repairs_stale_same_provider_reasoner_routes_when_reselecting_same_model():
    from cli.dev import _sync_routing_models_on_primary_switch

    cfg_data = {
        "model": "copilot/gpt-5.4-mini",
        "routing": {
            "reader": "bailian/qwen3.6-plus",
            "reasoner": "copilot/gpt-5.4",
            "complex": "copilot/o3",
            "repair": "bailian/qwen3.6-plus",
        },
    }

    changed = _sync_routing_models_on_primary_switch(
        cfg_data,
        old_model="copilot/gpt-5.4-mini",
        new_model="copilot/gpt-5.4-mini",
    )

    assert changed == ["reasoner", "complex"]
    assert cfg_data["routing"]["reader"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/gpt-5.4-mini"
    assert cfg_data["routing"]["complex"] == "copilot/gpt-5.4-mini"
    assert cfg_data["routing"]["repair"] == "bailian/qwen3.6-plus"


def test_dev_model_prefers_current_or_reasoning_model():
    from cli.dev import _preferred_model_index

    models = [
        {"id": "gpt-4.5"},
        {"id": "gpt-5.4-mini", "thinking": True},
        {"id": "o3", "reasoning": True},
    ]

    assert _preferred_model_index(models, current_model_id="o3") == 2
    assert _preferred_model_index(models, current_model_id="") == 1


def test_chat_reply_is_persisted_before_post_tick_cleanup():
    asyncio.run(_chat_reply_is_persisted_before_post_tick_cleanup())


async def _chat_reply_is_persisted_before_post_tick_cleanup():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            async def _sense(*args, **kwargs):
                return cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                Any,
                SimpleNamespace(
                    repeat_action_count=0,
                    repeat_action_tool="",
                    repeat_action_key="",
                    repeat_read_count=0,
                    repeat_read_path="",
                    loop_probe_version=0,
                ),
            )

            async def _decide(*args, **kwargs):
                return _judgment_output(
                    decision="pause",
                    rationale="已经找到根因",
                    reply_to_user="最终答复",
                )

            loop._judgment.decide = _decide
            loop._judgment._last_call_meta = {
                "model_ref": cfg.model,
                "thinking": cfg.thinking,
                "tier": "reasoner",
                "phase": "initial",
            }

            async def _boom(*args, **kwargs):
                raise RuntimeError("post tick cleanup failed")

            loop._post_tick_memory = _boom

            with pytest.raises(RuntimeError, match="post tick cleanup failed"):
                await loop._tick(1, user_message="你好", chat_id="chat-1")

            msgs = await loop.task_store.get_chat_messages_since(0, "chat-1")
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert msgs[0]["content"] == "最终答复"
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_local_chat_reply_is_persisted_for_default_channel():
    asyncio.run(_local_chat_reply_is_persisted_for_default_channel())


async def _local_chat_reply_is_persisted_for_default_channel():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            async def _sense(*args, **kwargs):
                return cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                Any,
                SimpleNamespace(
                    repeat_action_count=0,
                    repeat_action_tool="",
                    repeat_action_key="",
                    repeat_read_count=0,
                    repeat_read_path="",
                    loop_probe_version=0,
                ),
            )

            async def _decide(*args, **kwargs):
                return _judgment_output(
                    decision="pause",
                    rationale="已经得到结论",
                    reply_to_user="这是本地 chat 的回复",
                )

            loop._judgment.decide = _decide
            loop._judgment._last_call_meta = {
                "model_ref": cfg.model,
                "thinking": cfg.thinking,
                "tier": "reasoner",
                "phase": "initial",
            }

            reply = await loop._tick(1, user_message="你好", chat_id="")

            assert reply == "这是本地 chat 的回复"
            msgs = await loop.task_store.get_chat_messages_since(0)
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert msgs[0]["content"] == "这是本地 chat 的回复"
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_resolve_reply_chat_id_falls_back_to_last_chat_fact():
    asyncio.run(_resolve_reply_chat_id_falls_back_to_last_chat_fact())


async def _resolve_reply_chat_id_falls_back_to_last_chat_fact():
    from core.loop.chat import _resolve_reply_chat_id
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "chat-fallback.db")
        await store.open()
        try:
            await store.set_fact("chat:last_chat_id", "wechat:user-9", scope="system")
            loop = SimpleNamespace(_task_store=store)
            chat_id = await _resolve_reply_chat_id(loop, None, None)
            assert chat_id == "wechat:user-9"
            assert await _resolve_reply_chat_id(loop, None, "") == ""
        finally:
            await store.close()


def test_autonomous_followup_reply_uses_bound_chat_session():
    asyncio.run(_autonomous_followup_reply_uses_bound_chat_session())


async def _autonomous_followup_reply_uses_bound_chat_session():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg_path = Path.home() / ".lingzhou" / "lingzhou.json"
    if not cfg_path.exists():
        cfg_path = _proj_root() / "lingzhou.json.example"
    cfg = Config.load(cfg_path)
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            task_id = await loop.task_store.add_task(
                "继续向用户确认",
                goal="等待用户回复后继续",
                source="external",
                next_step="追问用户缺失信息",
            )
            await loop.task_store.update_status(task_id, "in_progress", "追问用户缺失信息")
            await loop.task_store.set_fact(f"task:{task_id}:chat_id", "wechat:user-1", scope="task")

            async def _sense(*args, **kwargs):
                return cast(Any, SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                Any,
                SimpleNamespace(
                    repeat_action_count=0,
                    repeat_action_tool="",
                    repeat_action_key="",
                    repeat_read_count=0,
                    repeat_read_path="",
                    loop_probe_version=0,
                ),
            )

            async def _decide(*args, **kwargs):
                return _judgment_output(
                    decision="pause",
                    rationale="需要用户补充一个关键参数",
                    reply_to_user="我还缺一个参数，麻烦补充一下。",
                )

            loop._judgment.decide = _decide
            loop._judgment._last_call_meta = {
                "model_ref": cfg.model,
                "thinking": cfg.thinking,
                "tier": "reasoner",
                "phase": "initial",
            }

            reply = await loop._tick(1)

            assert reply == "我还缺一个参数，麻烦补充一下。"
            msgs = await loop.task_store.get_chat_messages_since(0, "wechat:user-1")
            assert len(msgs) == 1
            assert msgs[0]["role"] == "assistant"
            assert msgs[0]["content"] == "我还缺一个参数，麻烦补充一下。"
        finally:
            await loop.task_store.close()
            await loop.provider.close()


