"""语义记忆（semantic）与情节记忆（episodic）测试"""
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
# SemanticMemory — Ebbinghaus 衰减
# ══════════════════════════════════════════════════════════════════════════════

def test_semantic_ebbinghaus():
    from memory.semantic import SemanticMemory, MemoryNode, effective_activation

    now_ts = datetime.now(UTC).isoformat()
    old_ts = (datetime.now(UTC) - timedelta(days=7)).isoformat()

    n_new = MemoryNode(id="new", kind="fact", title="python reload",
                       body="importlib", activation=0.8, created_at=now_ts)
    n_old = MemoryNode(id="old", kind="fact", title="python reload",
                       body="importlib", activation=0.8, created_at=old_ts)

    eff_new = effective_activation(n_new, 0.1)
    eff_old = effective_activation(n_old, 0.1)
    expected = 0.8 * math.exp(-0.1 * 7)

    assert eff_new > eff_old
    assert abs(eff_old - expected) < 0.01
    assert effective_activation(n_old, 0.0) == 0.8  # λ=0 不衰减

    with tempfile.TemporaryDirectory() as d:
        sm = SemanticMemory(Path(d), decay_lambda=0.1)
        sm.upsert(n_new)
        sm.upsert(n_old)
        results = sm.retrieve("python reload importlib", top_k=2)
        assert results[0]["id"] == "new"  # 新节点排前


# ══════════════════════════════════════════════════════════════════════════════
# EpisodicMemory — events.jsonl 轮转
# ══════════════════════════════════════════════════════════════════════════════

def test_episodic_rotation():
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=10)
        for i in range(20):
            ep.record_event("perception", {"seq": i})

        events = ep.list_events("perception", limit=100)
        assert len(events) <= 10
        assert events[-1]["seq"] == 19   # 最新
        assert events[0]["seq"] == 10    # 保留最新 10 条


def test_episodic_no_rotation():
    """max_events=0 时不做任何裁剪。"""
    from memory.episodic import EpisodicMemory

    with tempfile.TemporaryDirectory() as d:
        ep = EpisodicMemory(Path(d), max_events=0)
        for i in range(20):
            ep.record_event("perception", {"seq": i})
        events = ep.list_events("perception", limit=100)
        assert len(events) == 20


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap 注入
# ══════════════════════════════════════════════════════════════════════════════
