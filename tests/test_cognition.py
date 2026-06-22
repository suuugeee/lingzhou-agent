"""认知循环、chat reply、resolve 等集成测试"""
import asyncio
import json
import os
import sqlite3
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import (
    _judgment_output,
    _proj_root,
    _tool_registry,
)


def _continue_cfg(*, thresholds: dict[str, Any] | None = None, loop: dict[str, Any] | None = None):
    from core.config import Config

    data: dict[str, Any] = {
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
    }
    if thresholds is not None:
        data["thresholds"] = thresholds
    if loop:
        data["loop"] = loop
    return Config.model_validate(data)


class _ContinueStore:
    async def has_pending_chat_message(self) -> bool:
        return False


class _ContinueWorkbenchRegistry:
    def get(self, name: str):
        if name != "task.workbench":
            return None
        from tools.registry import ToolEntry, ToolManifest

        return ToolEntry(
            manifest=ToolManifest(name="task.workbench", description="demo"),
            handler=lambda params, ctx: None,  # type: ignore[arg-type]
        )


class _ContinueBehavior:
    def on_act(self, *args, **kwargs):
        return []

    def apply_cognitive_probe(self, signals):
        return None

    def apply_execution_gate(self, action, signals):
        return action

    def on_act_result(self, *args, **kwargs):
        return None


class _ContinueExecution:
    def __init__(self, *, workbench_summary: str = "") -> None:
        self.actions: list[Any] = []
        self.workbench_summary = workbench_summary

    async def dispatch(self, action, ctx):
        from tools.registry import ToolResult

        self.actions.append(action)
        if action.chosen_action_id == "task.workbench" and self.workbench_summary:
            return ToolResult(summary=self.workbench_summary)
        return ToolResult(summary=f"executed {action.chosen_action_id}")


def _tool_result(summary: str):
    from tools.registry import ToolResult

    return ToolResult(summary=summary)


def _continue_loop(
    *,
    cfg: Any,
    judgment: Any,
    execution: Any | None = None,
    registry: Any | None = None,
) -> SimpleNamespace:
    from memory.working import WorkingMemory

    return SimpleNamespace(
        _cfg=cfg,
        _emotion=SimpleNamespace(valence=0.5, arousal=0.5),
        _task_store=_ContinueStore(),
        _judgment=judgment,
        _pending_routing_overrides=None,
        _registry=registry or _tool_registry(),
        _wm=WorkingMemory(capacity=20),
        _execution=execution or _ContinueExecution(),
        _behavior=_ContinueBehavior(),
        _episodic=SimpleNamespace(record=lambda **kwargs: None),
        _bootstrap_mode="none",
    )


def test_bootstrap_wm_injection():
    from memory.working import WMItem, WorkingMemory

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


def test_soul_engine_renders_soul_md_without_hard_axioms():
    asyncio.run(_soul_engine_renders_soul_md_without_hard_axioms())


async def _soul_engine_renders_soul_md_without_hard_axioms():
    from core.persona import PersonaEngine
    from core.soul import SoulEngine
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        workspace = root / "workspace"
        workspace.mkdir()
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            await store.set_fact("soul:name", "lingzhou")
            await store.set_fact(
                "soul:ethos_baseline",
                json.dumps({
                    "truth": 0.8,
                    "caution": 0.7,
                    "continuity": 0.6,
                    "curiosity": 0.5,
                    "care": 0.4,
                }),
            )
            await store.set_fact("soul:hard_axioms", json.dumps(["do not render this axiom"]))

            cfg = SimpleNamespace(workspace_dir=workspace, soul=SimpleNamespace(name="lingzhou"))
            soul = SoulEngine(cfg, PersonaEngine(cfg, store))
            await soul.init_md()

            content = (workspace / "SOUL.md").read_text(encoding="utf-8")
            assert "do not render this axiom" not in content
            assert "CONSTITUTION.md" in content
            assert "constitution_hash" in content
            assert "真实 (truth):      0.800" in content
        finally:
            await store.close()


def test_extract_constitution_boundaries_prefers_absolute_boundary_section():
    from core.immune import extract_constitution_boundaries

    text = """
# CONSTITUTION.md

## 绝对边界（行动禁区）

1. 不欺骗用户
2. 不绕过人类监督机制

## 其他

- 不应出现在硬边界摘要
"""
    assert extract_constitution_boundaries(text) == [
        "不欺骗用户",
        "不绕过人类监督机制",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 完整构造链路（不调 LLM）
# ══════════════════════════════════════════════════════════════════════════════

def test_cognition_loop_init():
    """CognitionLoop.__init__ 不崩溃，关键参数正确传递。"""
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop
    from core.loop.runtime.context import RuntimeContext

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        assert isinstance(loop._runtime, RuntimeContext)
        assert loop._runtime._semantic is loop.semantic
        assert loop._runtime._episodic is loop.episodic
        assert loop._runtime._task_store is loop.task_store
        assert loop._runtime._tick_dispatcher is loop._tick_dispatcher
        assert loop._judgment._assembler._probe_manager is loop.probe_manager
        assert loop.semantic.decay_lambda == cfg.memory.semantic_decay_lambda
        assert loop.episodic.max_events == cfg.memory.max_events


def test_memory_config_uses_semantic_decay_lambda_only():
    from core.config_models import MemoryConfig

    cfg = MemoryConfig.model_validate({"semantic_decay_lambda": 0.25, "decay_lambda": 0.8})

    assert cfg.semantic_decay_lambda == 0.25
    assert not hasattr(cfg, "decay_lambda")


def test_curiosity_signal_does_not_auto_create_task():
    asyncio.run(_curiosity_signal_does_not_auto_create_task())


def test_curiosity_signal_skips_when_waiting_task_exists():
    asyncio.run(_curiosity_signal_skips_when_waiting_task_exists())


async def _curiosity_signal_does_not_auto_create_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
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
            ethos_state = cast("Any", SimpleNamespace(
                values=SimpleNamespace(curiosity=cfg.thresholds.curiosity_idle_task + 0.1)
            ))

            await loop._emit_curiosity_signal(ethos_state)

            tasks = await loop.task_store.list_tasks(limit=20)
            assert tasks == []
            # curiosity 仍然只注入感知信号，不直接创建任务；
            # 验证 _last_curiosity_signal_idle_cycle 已被标记（防止重复触发）
            assert loop._last_curiosity_signal_idle_cycle == loop._idle_cycles
        finally:
            await loop.task_store.close()
            await loop.provider.close()


async def _curiosity_signal_skips_when_waiting_task_exists():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            await loop.task_store.add_task(
                "等待用户反馈的任务",
                goal="等待同一个会话里的用户输入，不应触发 curiosity 空闲自唤醒",
                status="waiting",
                wait_kind="external",
                next_step="收到用户消息后继续",
            )
            loop._idle_cycles = cfg.thresholds.curiosity_idle_min_cycles
            loop._last_curiosity_signal_idle_cycle = 0
            ethos_state = cast("Any", SimpleNamespace(
                values=SimpleNamespace(curiosity=cfg.thresholds.curiosity_idle_task + 0.1)
            ))

            await loop._emit_curiosity_signal(ethos_state)

            assert loop._last_curiosity_signal_idle_cycle == 0
            assert [item for item in loop._wm.get_top(10) if item["kind"] == "curiosity"] == []
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_self_drive_signal_auto_creates_lightweight_growth_task():
    asyncio.run(_self_drive_signal_auto_creates_lightweight_growth_task())


def test_prepare_tick_adopts_auto_created_self_drive_task():
    asyncio.run(_prepare_tick_adopts_auto_created_self_drive_task())


def test_self_drive_signal_does_not_duplicate_pending_growth_task():
    asyncio.run(_self_drive_signal_does_not_duplicate_pending_growth_task())


def test_self_drive_signal_does_not_duplicate_waiting_growth_task():
    asyncio.run(_self_drive_signal_does_not_duplicate_waiting_growth_task())


def test_self_drive_signal_skips_when_waiting_external_task_exists():
    asyncio.run(_self_drive_signal_skips_when_waiting_external_task_exists())


def test_self_drive_signal_preserves_in_progress_focus_when_exploration_stuck():
    asyncio.run(_self_drive_signal_preserves_in_progress_focus_when_exploration_stuck())


def test_self_drive_feedback_receives_tick_event():
    from core.loop.tick.exec import _update_self_drive_from_tick
    from tools.registry import ToolResult

    class _SelfDrive:
        def __init__(self):
            self.events: list[dict[str, Any]] = []

        def update_from_tick(self, events: list[dict[str, Any]]) -> None:
            self.events.extend(events)

    self_drive = _SelfDrive()
    loop = SimpleNamespace(
        _self_drive=self_drive,
        _last_action_status="ok",
        _last_act_progressful=True,
    )
    action = _judgment_output(decision="act", chosen_action_id="task.complete", params={})
    replay = SimpleNamespace(avg_prediction_error=0.42)

    _update_self_drive_from_tick(loop, action, ToolResult(summary="任务已完成"), replay)

    assert self_drive.events == [{
        "type": "task_complete",
        "summary": "任务已完成",
        "status": "ok",
        "progressful": True,
        "prediction_error": 0.42,
    }]


def test_self_drive_engine_updates_from_tick_feedback(tmp_path):
    from core.loop.drive.engine import SelfDriveEngine

    engine = SelfDriveEngine(str(tmp_path / "runtime.db"))
    before = engine.snapshot()

    engine.update_from_tick([{
        "type": "task_complete",
        "summary": "完成一次自驱复盘",
        "prediction_error": 0.7,
    }])

    after = engine.snapshot()
    assert after["tasks_completed"] == before["tasks_completed"] + 1
    assert after["prediction_error_ema"] > before["prediction_error_ema"]

    template = engine.generate_exploration_task("memory_system")
    assert template["domain"] == "memory_system"
    assert template["question"]
    assert template["evidence_needed"]
    assert template["artifact"]
    assert template["done_condition"]
    assert "具体" in " ".join(template["evidence_needed"])

    for domain in (
        "code_structure",
        "tool_mastery",
        "memory_system",
        "self_evolution",
        "environment",
        "error_patterns",
        "api_integration",
        "performance",
    ):
        candidate = engine.generate_exploration_task(domain)
        combined = " ".join([
            str(candidate.get("goal") or ""),
            str(candidate.get("next_step") or ""),
            str(candidate.get("question") or ""),
            str(candidate.get("artifact") or ""),
            str(candidate.get("done_condition") or ""),
            " ".join(str(item) for item in candidate.get("evidence_needed") or []),
        ])
        assert candidate["question"]
        assert candidate["artifact"]
        assert candidate["evidence_needed"]
        assert candidate["done_condition"]
        assert any(marker in combined for marker in ("具体", "证据", "路径", "指标", "测试"))


def test_crash_recovery_uses_runtime_emotion_fallback():
    from core.loop.runtime.startup import _inject_crash_recovery
    from memory.working import WorkingMemory

    with tempfile.TemporaryDirectory() as d:
        state_dir = Path(d)
        (state_dir / "survival.json").write_text(
            json.dumps(
                {
                    "tick": 7,
                    "ts": "2026-05-22T10:00:00Z",
                    "active_task_title": "恢复任务",
                    "last_action": "task.resume",
                    "emotion": {},
                    "exit_type": "crash",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        wm = WorkingMemory(capacity=10)
        loop = SimpleNamespace(
            _cfg=SimpleNamespace(state_dir=state_dir),
            _wm=wm,
            _emotion=SimpleNamespace(valence=0.44, arousal=0.66),
        )

        _inject_crash_recovery(loop)

        items = wm.get_top(5)
        assert len(items) == 1
        assert items[0]["kind"] == "crash_recovery"
        assert "valence=0.44" in items[0]["content"]
        assert "arousal=0.66" in items[0]["content"]


def test_runtime_ready_callback_is_single_use():
    from core.loop.runtime.lifecycle import _invoke_runtime_ready_callback

    calls: list[str] = []
    loop = SimpleNamespace(_runtime_ready_callback=lambda: calls.append("ready"))

    _invoke_runtime_ready_callback(loop)
    _invoke_runtime_ready_callback(loop)

    assert calls == ["ready"]
    assert loop._runtime_ready_callback is None


def test_runtime_lifecycle_marks_clean_exit(tmp_path):
    from core.loop.runtime.lifecycle import _mark_clean_exit

    snapshot_path = tmp_path / "survival.json"
    snapshot_path.write_text(
        json.dumps({"tick": 3, "exit_type": "crash"}, ensure_ascii=False),
        encoding="utf-8",
    )
    loop = SimpleNamespace(_cfg=SimpleNamespace(state_dir=tmp_path))

    _mark_clean_exit(loop)

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["exit_type"] == "clean"


async def _self_drive_signal_auto_creates_lightweight_growth_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles

            await loop._emit_self_drive_signal()

            tasks = await loop.task_store.list_tasks(limit=20)
            self_drive_tasks = [task for task in tasks if task.source == "self_drive"]
            assert len(self_drive_tasks) == 1
            task = self_drive_tasks[0]
            assert task.status == "pending"
            assert task.next_step
            assert task.model_tier == "reasoner"
            assert task.result_json["cortex"]["intent"] == "self_drive_growth"
            assert task.extras["evidence_needed"]
            assert task.extras["artifact"]
            completion_checks = task.result_json["cortex"]["completion_checks"]
            assert any("产物已写入" in item for item in completion_checks)

            wm_items = loop._wm.get_top(10)
            self_drive_items = [item for item in wm_items if item["kind"] == "self_drive"]
            assert len(self_drive_items) == 1
            content = self_drive_items[0]["content"]
            assert "[自驱事件]" in content
            assert "scope: observation" in content
            assert f"created_self_drive_task: {task.id}" in content
            assert "proposal:" in content
            assert "candidate_question:" in content
            assert "candidate_evidence_needed:" in content
            assert "candidate_artifact:" in content
            assert "candidate_done_condition:" in content
            assert "open_questions:" in content
            assert "available_directions:" in content
            assert "task.add" not in content
        finally:
            await loop.task_store.close()
            await loop.provider.close()


async def _prepare_tick_adopts_auto_created_self_drive_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop
    from core.loop.tick import _prepare_active_task_for_tick

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles

            active_task = await _prepare_active_task_for_tick(loop, user_message="", chat_id=None)

            assert active_task is not None
            assert active_task.source == "self_drive"
            assert active_task.result_json["cortex"]["intent"] == "self_drive_growth"
            assert active_task.next_step
            anchors = [item for item in loop._wm.get_top(10) if item["kind"] == "task_anchor"]
            assert len(anchors) == 1
            assert active_task.title in anchors[0]["content"]
            assert active_task.next_step in anchors[0]["content"]
        finally:
            await loop.task_store.close()
            await loop.provider.close()


async def _self_drive_signal_does_not_duplicate_pending_growth_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            existing_id = await loop.task_store.add_task(
                "已有自驱成长任务",
                goal="避免重复创建",
                source="self_drive",
                status="pending",
                next_step="继续已有成长任务",
            )
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles

            await loop._emit_self_drive_signal()

            tasks = await loop.task_store.list_tasks(limit=20)
            self_drive_tasks = [task for task in tasks if task.source == "self_drive"]
            assert [task.id for task in self_drive_tasks] == [existing_id]

            wm_items = loop._wm.get_top(10)
            self_drive_items = [item for item in wm_items if item["kind"] == "self_drive"]
            assert self_drive_items == []
        finally:
            await loop.task_store.close()
            await loop.provider.close()


async def _self_drive_signal_does_not_duplicate_waiting_growth_task():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            existing_id = await loop.task_store.add_task(
                "已有等待中的自驱成长任务",
                goal="等待中的自驱任务也应阻止重复创建",
                source="self_drive",
                status="waiting",
                wait_kind="external",
                next_step="收到外部日志后继续取证",
            )
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles

            await loop._emit_self_drive_signal()

            tasks = await loop.task_store.list_tasks(limit=20)
            self_drive_tasks = [task for task in tasks if task.source == "self_drive"]
            assert [task.id for task in self_drive_tasks] == [existing_id]
            assert [item for item in loop._wm.get_top(10) if item["kind"] == "self_drive"] == []
        finally:
            await loop.task_store.close()
            await loop.provider.close()


async def _self_drive_signal_skips_when_waiting_external_task_exists():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop
    from core.loop.cycle.focus import claim_focus_task

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            existing_id = await loop.task_store.add_task(
                "等待用户输入的普通任务",
                goal="waiting 普通任务也应阻止 self_drive 抢跑",
                source="external",
                status="waiting",
                wait_kind="external",
                next_step="收到用户消息后继续",
            )
            task = await loop.task_store.get_task_by_id(existing_id)
            await claim_focus_task(loop, task, clear_current=True)
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles

            await loop._emit_self_drive_signal()

            tasks = await loop.task_store.list_tasks(limit=20)
            assert [task.id for task in tasks if task.source == "self_drive"] == []
            assert [item for item in loop._wm.get_top(10) if item["kind"] == "self_drive"] == []
        finally:
            await loop.task_store.close()
            await loop.provider.close()


async def _self_drive_signal_preserves_in_progress_focus_when_exploration_stuck():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop
    from core.loop.cycle.focus import claim_focus_task

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        await loop.task_store.open()
        try:
            existing_id = await loop.task_store.add_task(
                "已有进行中自驱任务",
                goal="保持当前焦点，不再创建新探索",
                source="self_drive",
                status="in_progress",
                next_step="继续读取当前关键链路",
            )
            existing = await loop.task_store.get_task_by_id(existing_id)
            assert existing is not None
            await claim_focus_task(loop, existing, clear_current=True)
            loop._behavior._wait_streak = cfg.thresholds.curiosity_idle_min_cycles
            loop._behavior._read_streak_count = cfg.loop.behavior_streak_threshold + 2

            await loop._emit_self_drive_signal()

            tasks = await loop.task_store.list_tasks(limit=20)
            self_drive_tasks = [task for task in tasks if task.source == "self_drive"]
            assert [task.id for task in self_drive_tasks] == [existing_id]
            assert [item for item in loop._wm.get_top(10) if item["kind"] == "self_drive"] == []
        finally:
            await loop.task_store.close()
            await loop.provider.close()


def test_self_drive_signal_bypasses_idle_judge_aggregation():
    asyncio.run(_self_drive_signal_bypasses_idle_judge_aggregation())


def test_continue_phase_uses_configured_tool_history_compaction_threshold():
    asyncio.run(_continue_phase_uses_configured_tool_history_compaction_threshold())


def test_continue_phase_records_workbench_when_inner_round_limit_reached():
    asyncio.run(_continue_phase_records_workbench_when_inner_round_limit_reached())


def test_continue_phase_default_disables_inner_round_limit():
    asyncio.run(_continue_phase_default_disables_inner_round_limit())


def test_continue_phase_gates_repeated_same_action_before_dispatch():
    asyncio.run(_continue_phase_gates_repeated_same_action_before_dispatch())


def test_idle_decision_phase_blocks_taskless_autonomous_act(monkeypatch: pytest.MonkeyPatch):
    asyncio.run(_idle_decision_phase_blocks_taskless_autonomous_act(monkeypatch))


def test_continue_phase_inherits_task_id_for_task_scoped_tools():
    asyncio.run(_continue_phase_inherits_task_id_for_task_scoped_tools())


async def _continue_phase_uses_configured_tool_history_compaction_threshold():
    from core.loop.shared.continue_phase import _run_continue_phase

    cfg = _continue_cfg(
        thresholds={
            "continue_tool_history_compact_threshold": 2,
            "continue_tool_history_keep_last": 1,
        },
    )

    seen_histories: list[list[dict[str, Any]]] = []

    class _Judgment:
        async def decide_continue(self, tool_history, **kwargs):
            seen_histories.append([dict(item) for item in tool_history])
            return _judgment_output(decision="wait", rationale="证据已足够")

    loop = _continue_loop(cfg=cfg, judgment=_Judgment())

    tool_history = [
        {
            "tool": "memory.search",
            "params": {"query": "legacy runtime"},
            "result": "命中 1 条",
            "status": "ok",
            "error": "",
            "summary": "命中 1 条",
            "state_delta": {},
        },
        {
            "tool": "task.list",
            "params": {"status": "all"},
            "result": "命中 1 条任务",
            "status": "ok",
            "error": "",
            "summary": "命中 1 条任务",
            "state_delta": {},
        },
    ]

    await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="memory.search", params={"query": "legacy runtime"}),
        result=_tool_result("初始结果"),
        tool_history=tool_history,
    )

    assert len(seen_histories) == 1
    assert seen_histories[0][0]["tool"] == "[compacted]"
    assert "早期 1 条工具调用已" in seen_histories[0][0]["result"]
    assert "压缩" in seen_histories[0][0]["result"]
    assert seen_histories[0][1]["tool"] == "task.list"


async def _continue_phase_records_workbench_when_inner_round_limit_reached():
    from core.loop.shared.continue_phase import _run_continue_phase

    cfg = _continue_cfg(
        thresholds={
            "continue_max_inner_rounds": 2,
            "continue_tool_history_compact_threshold": 20,
            "continue_tool_history_keep_last": 10,
        },
    )

    class _Judgment:
        def __init__(self) -> None:
            self.calls = 0

        async def decide_continue(self, tool_history, **kwargs):
            self.calls += 1
            return _judgment_output(
                decision="act",
                chosen_action_id="file.read",
                params={"path": f"/tmp/{self.calls}.txt"},
                rationale="继续读取",
            )

    judgment = _Judgment()
    execution = _ContinueExecution(workbench_summary="工作台已更新")
    loop = _continue_loop(
        cfg=cfg,
        judgment=judgment,
        execution=execution,
        registry=_ContinueWorkbenchRegistry(),
    )

    final_action, final_result = await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/tmp/start.txt"}),
        result=_tool_result("初始结果"),
        tool_history=[],
    )

    assert judgment.calls == 2
    assert final_action.chosen_action_id == "task.workbench"
    assert final_result.summary == "工作台已更新"
    assert execution.actions[-1].chosen_action_id == "task.workbench"
    workbench = execution.actions[-1].params["workbench"]
    assert workbench["recovery_state"] == "continue_round_limit_reached"
    assert "本 tick continue 阶段已执行 2 轮工具续判" in workbench["evidence"][0]
    assert "显式 continue 轮次上限" in workbench["next_verification"]
    assert workbench["verification_state"]["status"] == "resolved"
    assert "建议的后续验证入口" in workbench["evidence"][2]


async def _continue_phase_default_disables_inner_round_limit():
    from core.loop.shared.continue_phase import _run_continue_phase

    cfg = _continue_cfg(thresholds=None)

    class _Judgment:
        def __init__(self) -> None:
            self.calls = 0

        async def decide_continue(self, tool_history, **kwargs):
            self.calls += 1
            return _judgment_output(
                decision="act",
                chosen_action_id="file.read",
                params={"path": f"/tmp/default-{self.calls}.txt"},
                rationale="继续读取",
            )

    judgment = _Judgment()
    execution = _ContinueExecution(workbench_summary="默认低信息收敛工作台已更新")
    loop = _continue_loop(
        cfg=cfg,
        judgment=judgment,
        execution=execution,
        registry=_ContinueWorkbenchRegistry(),
    )

    final_action, final_result = await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/tmp/start.txt"}),
        result=_tool_result("初始结果"),
        tool_history=[],
    )

    assert cfg.thresholds.continue_max_inner_rounds == 0
    assert judgment.calls == 3
    assert final_action.chosen_action_id == "task.workbench"
    assert final_result.summary == "默认低信息收敛工作台已更新"
    assert execution.actions[-1].params["workbench"]["recovery_state"] == "continue_low_increment_budget_reached"


def test_continue_round_limit_prefers_runtime_recovery_next_step():
    from core.loop.shared.continue_phase import _specific_round_limit_next_verification

    next_verification = _specific_round_limit_next_verification([
        {
            "tool": "shell.run",
            "status": "skipped",
            "error": "ToolInputInvalid",
            "summary": "shell.run missing_params=command",
            "state_delta": {
                "tool_input_invalid": True,
                "missing_params": ["command"],
                "recovery_next_step": "按 shell.run 的 manifest 重新调用工具；补齐必填参数 command。",
            },
        }
    ])

    assert next_verification == "按 shell.run 的 manifest 重新调用工具；补齐必填参数 command。"


def test_continue_round_limit_ignores_control_text_when_deriving_next_verification():
    from core.cortex import intent as cortex_intent
    from core.loop.shared.continue_phase import _specific_round_limit_next_verification

    next_verification = _specific_round_limit_next_verification([
        {
            "tool": "task.workbench",
            "status": "succeeded",
            "state_delta": {
                "next_verification": cortex_intent.control_next_verification(
                    "下一轮先综合本 tick 工具结果，确认是否已经足够回答/完成；"
                    "若不足，再选择一个最高信息增量的验证动作。"
                ),
            },
        },
        {
            "tool": "shell.run",
            "status": "succeeded",
            "state_delta": {
                "recovery_next_step": "按 shell.run 的 manifest 重新调用工具；补齐必填参数 command。",
            },
        },
    ])

    assert next_verification == "按 shell.run 的 manifest 重新调用工具；补齐必填参数 command。"


async def _continue_phase_gates_repeated_same_action_before_dispatch():
    from core.loop.shared.continue_phase import _run_continue_phase

    cfg = _continue_cfg(
        loop={"behavior_streak_threshold": 3},
        thresholds={
            "continue_max_inner_rounds": 4,
            "continue_tool_history_compact_threshold": 20,
            "continue_tool_history_keep_last": 10,
        },
    )

    class _Judgment:
        async def decide_continue(self, tool_history, **kwargs):
            return _judgment_output(
                decision="act",
                chosen_action_id="file.read",
                params={"path": "/tmp/repeat.py"},
                rationale="继续读取同一文件",
            )

    execution = _ContinueExecution()
    loop = _continue_loop(
        cfg=cfg,
        judgment=_Judgment(),
        execution=execution,
        registry=_ContinueWorkbenchRegistry(),
    )
    tool_history = [
        {
            "tool": "file.read",
            "params": {"path": "/tmp/repeat.py"},
            "result": "same",
            "status": "ok",
            "error": "",
            "summary": "same",
            "state_delta": {},
        },
        {
            "tool": "file.read",
            "params": {"path": "/tmp/repeat.py"},
            "result": "same",
            "status": "ok",
            "error": "",
            "summary": "same",
            "state_delta": {},
        },
    ]

    final_action, final_result = await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="用户正在等结论",
        active_task=SimpleNamespace(id=1),
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="file.read", params={"path": "/tmp/repeat.py"}),
        result=_tool_result("初始结果"),
        tool_history=tool_history,
    )

    assert final_action.chosen_action_id == "task.workbench"
    assert final_result.summary == "executed task.workbench"
    assert len(execution.actions) == 1
    assert execution.actions[0].chosen_action_id == "task.workbench"
    workbench = execution.actions[0].params["workbench"]
    assert workbench["recovery_state"] == "continue_repeat_action_gated"
    assert "/tmp/repeat.py" in workbench["next_verification"]
    assert "shell.run/grep" in workbench["next_verification"]


async def _idle_decision_phase_blocks_taskless_autonomous_act(monkeypatch: pytest.MonkeyPatch):
    import core.loop.tick.prep as prep_module

    async def _fake_decide(*args, **kwargs):
        return _judgment_output(
            decision="act",
            chosen_action_id="memory.search",
            params={"query": "OpenClaw"},
            rationale="waiting 任务的下一步要求继续取证",
        )

    async def _fake_review(loop, ctx, action, user_message, active_task):
        return action

    async def _maybe_curiosity_task(*args, **kwargs):
        return None

    monkeypatch.setattr(prep_module, "_decide_initial_action", _fake_decide)
    monkeypatch.setattr(prep_module, "_review_delegate_tasks", _fake_review)
    monkeypatch.setattr(prep_module, "_log_tick_decision", lambda *args, **kwargs: None)

    loop = SimpleNamespace(_maybe_curiosity_task=_maybe_curiosity_task)
    prep = SimpleNamespace(ethos_state=SimpleNamespace())

    blocked = await prep_module._TickJudgmentPhase.run(
        loop,
        SimpleNamespace(),
        1,
        "",
        None,
        None,
        prep,
    )
    allowed = await prep_module._TickJudgmentPhase.run(
        loop,
        SimpleNamespace(),
        1,
        "",
        SimpleNamespace(id=1, extras={}),
        None,
        prep,
    )

    assert blocked.decision == "wait"
    assert "没有活跃任务" in blocked.rationale
    assert allowed.decision == "act"
    assert allowed.chosen_action_id == "memory.search"


async def _continue_phase_inherits_task_id_for_task_scoped_tools():
    from core.loop.shared.continue_phase import _run_continue_phase

    cfg = _continue_cfg(
        thresholds={
            "continue_max_inner_rounds": 2,
            "continue_tool_history_compact_threshold": 20,
            "continue_tool_history_keep_last": 10,
        },
    )

    class _Judgment:
        def __init__(self) -> None:
            self.calls = 0

        async def decide_continue(self, tool_history, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _judgment_output(
                    decision="act",
                    chosen_action_id="task.workbench",
                    params={"workbench": {"evidence": ["done"]}},
                    rationale="写入工作台",
                )
            return _judgment_output(decision="wait", rationale="done")

    execution = _ContinueExecution()
    loop = _continue_loop(
        cfg=cfg,
        judgment=_Judgment(),
        execution=execution,
        registry=_ContinueWorkbenchRegistry(),
    )
    tool_history = [
        {
            "tool": "task.add",
            "params": {"title": "new task"},
            "result": "task.add id=3794",
            "status": "ok",
            "error": "",
            "summary": "task.add id=3794",
            "resource_key": "3794",
            "metadata": {"task_id": 3794},
            "state_delta": {"task_status": "pending"},
        }
    ]

    await _run_continue_phase(
        loop=loop,
        ctx=SimpleNamespace(),
        user_message="",
        active_task=None,
        cognitive_signals=SimpleNamespace(),
        action=_judgment_output(decision="act", chosen_action_id="task.add", params={"title": "new task"}),
        result=_tool_result("task.add id=3794"),
        tool_history=tool_history,
    )

    assert execution.actions[0].chosen_action_id == "task.workbench"
    assert execution.actions[0].params["task_id"] == 3794


async def _self_drive_signal_bypasses_idle_judge_aggregation():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.judgment import JudgmentOutput
    from core.loop import CognitionLoop
    from core.loop.tick import _decide_initial_action, _TickJudgmentPrep
    from memory.working import WMItem

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
    with tempfile.TemporaryDirectory() as d:
        cfg.loop.db_path = f"{d}/state/runtime.db"
        cfg.loop.memory_dir = f"{d}/memory"
        cfg.loop.workspace_dir = f"{d}/workspace"
        cfg.loop.act = False
        cfg.loop.judge_every = 3
        cfg.evolution.enabled = False

        loop = CognitionLoop(cfg)
        called = {"value": False}

        async def _fake_decide(*args, **kwargs):
            called["value"] = True
            return JudgmentOutput.wait(reason="llm saw self_drive")

        loop._judgment.decide = _fake_decide  # type: ignore[method-assign]
        loop._wm.add(WMItem(kind="self_drive", content="[自驱信号] test", priority=0.9))

        try:
            action = await _decide_initial_action(
                loop,
                cycle=1,
                user_message="",
                active_task=None,
                chat_id=None,
                prep=_TickJudgmentPrep(
                    percept=None,
                    perception_replay=None,
                    cognitive_signals=None,
                    ethos_state=None,
                    signals=None,
                    hard_boundaries=[],
                ),
            )

            assert called["value"] is True
            assert action.rationale == "llm saw self_drive"
            assert loop._ticks_since_judge == 0
        finally:
            await loop.provider.close()


def test_short_continuation_message_is_also_forwarded_to_active_task_inbox():
    from core.loop.tick import _should_steer_active_task_from_user_message
    from store.task import Task

    task = Task(
        id=1,
        title="继续分析 chat 回复",
        status="in_progress",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="继续分析 chat 回复为什么丢失",
        next_step="继续分析这个问题",
    )

    assert _should_steer_active_task_from_user_message(task, "继续分析") is True


def test_task_steer_any_nonempty_user_message_is_forwarded_to_active_task_inbox():
    from core.loop.tick import _should_steer_active_task_from_user_message
    from store.task import Task

    task = Task(
        id=1,
        title="alpha beta",
        status="in_progress",
        priority="high",
        created_at="2026-05-15T00:00:00+00:00",
        goal="alpha beta",
        next_step="alpha beta",
    )

    message = "alpha beta gamma delta"
    assert _should_steer_active_task_from_user_message(task, message) is True
    assert _should_steer_active_task_from_user_message(task, "继续分析") is True
    assert _should_steer_active_task_from_user_message(task, "   ") is False


def test_distinct_user_message_is_queued_into_active_task_inbox():
    asyncio.run(_distinct_user_message_is_queued_into_active_task_inbox())


async def _distinct_user_message_is_queued_into_active_task_inbox():
    from core.loop.tick import _maybe_steer_active_task_from_user_message
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "steer-user-message.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "旧回填任务",
                goal="等待模型加载完成后，再次检查日志确认数据回填进度",
                next_step="继续检查回填进度",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            updated = await _maybe_steer_active_task_from_user_message(
                store,
                task,
                "请你使用 puppeteer 去搜索。",
            )

            assert updated is not None
            assert len(updated.extras["inbox_messages"]) == 1
            assert updated.extras["inbox_messages"][0] == "收到新的用户消息：请你使用 puppeteer 去搜索。"

            persisted = await store.get_task_by_id(task_id)
            assert persisted is not None
            assert persisted.extras["inbox_messages"] == updated.extras["inbox_messages"]
        finally:
            await store.close()


def test_distinct_execute_user_message_is_persisted_into_action_first_cortex():
    asyncio.run(_distinct_execute_user_message_is_persisted_into_action_first_cortex())


async def _distinct_execute_user_message_is_persisted_into_action_first_cortex():
    from core.loop.tick import _maybe_steer_active_task_from_user_message
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "steer-action-first.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "处理代理配置",
                goal="应用用户给出的配置并验证网络链路",
                next_step="等待用户提供订阅地址",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None

            await _maybe_steer_active_task_from_user_message(
                store,
                task,
                "用这个url的配置 https://example.com/sub?clash=1，下载后测试",
            )

            persisted = await store.get_task_by_id(task_id)
            assert persisted is not None
            cortex = persisted.result_json["cortex"]
            assert cortex["action_first"]["intent"] == "execute"
            assert cortex["action_first"]["must_act"] is True
            assert {"kind": "url", "value": "https://example.com/sub?clash=1"} in cortex["captured_inputs"]
        finally:
            await store.close()


def test_invalid_routing_overrides_clear_previous_pending_state():
    asyncio.run(_invalid_routing_overrides_clear_previous_pending_state())


async def _invalid_routing_overrides_clear_previous_pending_state():
    from core.loop.tick import _apply_tick_model_strategy
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "routing-overrides.db")
        await store.open()
        try:
            await store.set_fact("pref:routing_overrides", '{"reader":"demo/model"}', scope="system")
            loop = cast("Any", SimpleNamespace(
                _cfg=SimpleNamespace(
                    loop=SimpleNamespace(
                        idle_with_task_bounds=[100, 30000],
                        idle_no_task_bounds=[5000, 300000],
                    )
                ),
                _task_store=store,
                _pending_routing_overrides={"reader": "demo/model"},
                _pending_tier="reader",
                _pending_idle_gap=1.0,
                _pending_thinking_override="low",
            ))
            action = _judgment_output(
                decision="wait",
                model_strategy={
                    "routing_overrides": {
                        "invalid": "demo/ignored",
                        "reader": "",
                    }
                },
            )

            await _apply_tick_model_strategy(loop, action, None)

            stored_value, found = await store.get_fact("pref:routing_overrides")
            assert loop._pending_routing_overrides is None
            assert found is True
            assert stored_value == ""
        finally:
            await store.close()


def test_run_tick_maintenance_uses_configured_global_md_warn_lines():
    asyncio.run(_run_tick_maintenance_uses_configured_global_md_warn_lines())


def test_run_tick_maintenance_uses_configured_low_pressure_skip_threshold():
    asyncio.run(_run_tick_maintenance_uses_configured_low_pressure_skip_threshold())


def test_run_tick_maintenance_auto_compacts_large_runtime_and_memory():
    asyncio.run(_run_tick_maintenance_auto_compacts_large_runtime_and_memory())


def test_run_tick_maintenance_auto_compaction_records_errors(monkeypatch):
    asyncio.run(_run_tick_maintenance_auto_compaction_records_errors(monkeypatch))


def test_run_tick_maintenance_auto_compaction_skips_concurrent_run(monkeypatch):
    asyncio.run(_run_tick_maintenance_auto_compaction_skips_concurrent_run(monkeypatch))


async def _run_tick_maintenance_uses_configured_global_md_warn_lines():
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance

    class _WM:
        def __init__(self) -> None:
            self.pressure = 1.0
            self.items: list[Any] = []

        def add(self, item: Any) -> None:
            self.items.append(item)

    class _Soul:
        def __init__(self) -> None:
            self.synced = False

        async def sync_md(self) -> None:
            self.synced = True

    class _DB:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def execute(self, sql: str) -> None:
            self.commands.append(sql)

    with tempfile.TemporaryDirectory() as d:
        memory_dir = Path(d) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "episodic").mkdir(parents=True, exist_ok=True)
        (memory_dir / "episodic" / "global.md").write_text("a\nb\nc\n", encoding="utf-8")

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
                "memory_dir": str(memory_dir),
                "consolidate_every": 1,
            },
            "memory": {
                "global_md_warn_bytes": 999999,
                "global_md_warn_lines": 2,
            },
            "thresholds": {
                "wm_pressure_task": 0.8,
            },
        })

        db = _DB()
        soul = _Soul()
        wm = _WM()

        async def _wal_checkpoint() -> None:
            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        loop = cast("Any", SimpleNamespace(
            _cfg=cfg,
            _wm=wm,
            _soul=soul,
            _task_store=SimpleNamespace(_db=db, wal_checkpoint=_wal_checkpoint),
            consolidated=False,
        ))

        async def _consolidate(active_task: Any) -> None:
            loop.consolidated = True

        loop._consolidate = _consolidate

        await _run_tick_maintenance(loop, active_task=None, cycle=1)

        assert loop.consolidated is True
        assert soul.synced is True
        assert db.commands == ["PRAGMA wal_checkpoint(TRUNCATE)"]
        assert any("global.md 当前 3 行" in item.content for item in wm.items)


async def _run_tick_maintenance_uses_configured_low_pressure_skip_threshold():
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance

    class _WM:
        def __init__(self, pressure: float) -> None:
            self.pressure = pressure

    class _Soul:
        def __init__(self) -> None:
            self.synced = False

        async def sync_md(self) -> None:
            self.synced = True

    class _DB:
        async def execute(self, sql: str) -> None:
            return None

    async def _run_case(skip_threshold: float) -> bool:
        with tempfile.TemporaryDirectory() as d:
            memory_dir = Path(d) / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
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
                    "memory_dir": str(memory_dir),
                    "consolidate_every": 1,
                },
                "memory": {
                    "consolidate_low_pressure_skip_threshold": skip_threshold,
                    "global_md_warn_bytes": 999999,
                    "global_md_warn_lines": 999999,
                },
                "thresholds": {
                    "wm_pressure_task": 0.95,
                },
            })

            loop = cast("Any", SimpleNamespace(
                _cfg=cfg,
                _wm=_WM(pressure=0.86),
                _soul=_Soul(),
                _task_store=SimpleNamespace(_db=_DB()),
                consolidated=False,
            ))

            async def _consolidate(active_task: Any) -> None:
                loop.consolidated = True

            loop._consolidate = _consolidate
            await _run_tick_maintenance(loop, active_task=None, cycle=1)
            return bool(loop.consolidated)

    assert await _run_case(0.90) is False
    assert await _run_case(0.80) is True


async def _run_tick_maintenance_auto_compacts_large_runtime_and_memory():
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance

    class _WM:
        def __init__(self) -> None:
            self.pressure = 0.1
            self.items: list[Any] = []

        def add(self, item: Any) -> None:
            self.items.append(item)

    class _Soul:
        async def sync_md(self) -> None:
            return None

    class _TaskStore:
        def __init__(self) -> None:
            self.facts: dict[str, str] = {}

        async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
            _ = scope
            self.facts[key] = value

        async def wal_checkpoint(self) -> None:
            return None

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        db_path = root / "runtime.db"
        memory_dir = root / "memory"
        node_dir = memory_dir / "nodes"
        node_dir.mkdir(parents=True)
        huge = "M" * 20_000 + "TAIL"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}')")
            conn.execute(
                "INSERT INTO runs (id, data) VALUES (?, ?)",
                (1, json.dumps({"output_json": {"summary": huge}}, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()
        (node_dir / "large.json").write_text(
            json.dumps({"id": "large", "kind": "fact", "title": "large", "body": huge}, ensure_ascii=False),
            encoding="utf-8",
        )

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
                "db_path": str(db_path),
                "memory_dir": str(memory_dir),
                "consolidate_every": 100,
            },
            "memory": {
                "auto_compact_enabled": True,
                "auto_compact_every_ticks": 1,
                "auto_compact_runtime_db_min_bytes": 1,
                "auto_compact_memory_dir_min_bytes": 1,
                "auto_compact_vacuum": False,
                "global_md_warn_bytes": 999999,
                "global_md_warn_lines": 999999,
            },
            "thresholds": {
                "wm_pressure_task": 0.8,
            },
        })
        store = _TaskStore()
        loop = cast("Any", SimpleNamespace(
            _cfg=cfg,
            _wm=_WM(),
            _soul=_Soul(),
            _task_store=store,
            consolidated=False,
        ))

        async def _consolidate(active_task: Any) -> None:
            loop.consolidated = True

        loop._consolidate = _consolidate

        await _run_tick_maintenance(loop, active_task=None, cycle=1)

        conn = sqlite3.connect(db_path)
        try:
            run_data = json.loads(conn.execute("SELECT data FROM runs WHERE id=1").fetchone()[0])
        finally:
            conn.close()
        node_data = json.loads((node_dir / "large.json").read_text(encoding="utf-8"))
        marker = json.loads(store.facts["maintenance:auto_compact:last"])

        assert "persistent storage truncated" in run_data["output_json"]["summary"]
        assert "persistent storage truncated" in node_data["body"]
        assert {report["kind"] for report in marker["reports"]} == {"runtime_db", "memory_dir"}
        assert any("自动记忆维护" in item.content for item in loop._wm.items)
        assert loop.consolidated is False


async def _run_tick_maintenance_auto_compaction_records_errors(monkeypatch):
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance
    import core.maintenance as maintenance_mod

    def _raise_runtime_compaction(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = args, kwargs
        raise RuntimeError("database is locked")

    monkeypatch.setattr(maintenance_mod, "compact_runtime_db", _raise_runtime_compaction)

    class _WM:
        pressure = 0.1

        def __init__(self) -> None:
            self.items: list[Any] = []

        def add(self, item: Any) -> None:
            self.items.append(item)

    class _Soul:
        async def sync_md(self) -> None:
            return None

    class _TaskStore:
        def __init__(self) -> None:
            self.facts: dict[str, str] = {}

        async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
            _ = scope
            self.facts[key] = value

        async def wal_checkpoint(self) -> None:
            return None

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "runtime.db"
        db_path.write_text("large enough", encoding="utf-8")
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
                "db_path": str(db_path),
                "memory_dir": str(Path(d) / "memory"),
                "consolidate_every": 100,
            },
            "memory": {
                "auto_compact_enabled": True,
                "auto_compact_every_ticks": 1,
                "auto_compact_runtime_db_min_bytes": 1,
                "auto_compact_memory_dir_min_bytes": 0,
                "global_md_warn_bytes": 999999,
                "global_md_warn_lines": 999999,
            },
            "thresholds": {
                "wm_pressure_task": 0.8,
            },
        })
        store = _TaskStore()
        loop = cast("Any", SimpleNamespace(
            _cfg=cfg,
            _wm=_WM(),
            _soul=_Soul(),
            _task_store=store,
            consolidated=False,
        ))

        async def _consolidate(active_task: Any) -> None:
            loop.consolidated = True

        loop._consolidate = _consolidate

        await _run_tick_maintenance(loop, active_task=None, cycle=1)

        marker = json.loads(store.facts["maintenance:auto_compact:last"])
        assert marker["reports"][0]["kind"] == "runtime_db"
        assert marker["reports"][0]["error"] == "RuntimeError: database is locked"
        assert loop._wm.items == []
        assert loop.consolidated is False


async def _run_tick_maintenance_auto_compaction_skips_concurrent_run(monkeypatch):
    from core.config import Config
    from core.loop.tick import _run_tick_maintenance
    import core.maintenance as maintenance_mod

    calls = 0

    def _slow_runtime_compaction(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        _ = args, kwargs
        calls += 1
        time.sleep(0.05)
        return {
            "changed_rows": 1,
            "changed_files": 0,
            "saved_bytes": 10,
            "vacuumed": False,
        }

    monkeypatch.setattr(maintenance_mod, "compact_runtime_db", _slow_runtime_compaction)

    class _WM:
        pressure = 0.1

        def __init__(self) -> None:
            self.items: list[Any] = []

        def add(self, item: Any) -> None:
            self.items.append(item)

    class _Soul:
        async def sync_md(self) -> None:
            return None

    class _TaskStore:
        def __init__(self) -> None:
            self.facts: dict[str, str] = {}

        async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
            _ = scope
            self.facts[key] = value

        async def wal_checkpoint(self) -> None:
            return None

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "runtime.db"
        db_path.write_text("large enough", encoding="utf-8")
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
                "db_path": str(db_path),
                "memory_dir": str(Path(d) / "memory"),
                "consolidate_every": 100,
            },
            "memory": {
                "auto_compact_enabled": True,
                "auto_compact_every_ticks": 1,
                "auto_compact_runtime_db_min_bytes": 1,
                "auto_compact_memory_dir_min_bytes": 0,
                "global_md_warn_bytes": 999999,
                "global_md_warn_lines": 999999,
            },
            "thresholds": {
                "wm_pressure_task": 0.8,
            },
        })
        loop = cast("Any", SimpleNamespace(
            _cfg=cfg,
            _wm=_WM(),
            _soul=_Soul(),
            _task_store=_TaskStore(),
            consolidated=False,
        ))

        async def _consolidate(active_task: Any) -> None:
            loop.consolidated = True

        loop._consolidate = _consolidate

        await asyncio.gather(
            _run_tick_maintenance(loop, active_task=None, cycle=1),
            _run_tick_maintenance(loop, active_task=None, cycle=1),
        )

        assert calls == 1
        assert len(loop._wm.items) == 1
        assert "自动记忆维护" in loop._wm.items[0].content
        assert loop.consolidated is False


def test_post_tick_memory_crystallizes_task_summary_title_with_task_id():
    asyncio.run(_post_tick_memory_crystallizes_task_summary_title_with_task_id())


async def _post_tick_memory_crystallizes_task_summary_title_with_task_id():
    from core.loop.tick import _post_tick_memory_impl
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("重复标题任务", goal="crystallize summary", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            await store.update_status(task_id, "done", "finished")
            semantic = SemanticMemory(root / "semantic")
            loop = cast("Any", SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, n_recent=40000: "任务完成叙事"),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=3),
                    emotion=SimpleNamespace(),
                ),
            ))
            action = cast("Any", SimpleNamespace(chosen_action_id="", params={}, reflection="", rationale=""))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(loop, action, result, active_task, cycle=1, user_message="")

            node = semantic.get(f"task_summary_{task_id}")
            assert node is not None
            assert node.title == f"[done] task#{task_id} 重复标题任务"
        finally:
            await store.close()


def test_consolidate_promotes_semantic_nodes_and_durable_user_facts():
    asyncio.run(_consolidate_promotes_semantic_nodes_and_durable_user_facts())


async def _consolidate_promotes_semantic_nodes_and_durable_user_facts():
    from core.config_models import MemoryConfig
    from core.loop.runtime import CognitionLoop
    from memory.working import WMItem, WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        memory_dir = root / "memory"
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("记忆巩固测试", goal="consolidate memory", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None

            loop = cast("Any", SimpleNamespace(
                _wm=WorkingMemory(capacity=20),
                _episodic=EpisodicMemory(memory_dir),
                _semantic=SemanticMemory(memory_dir, decay_lambda=0.0),
                _task_store=store,
                _perception=SimpleNamespace(reset_wm_baseline=lambda size: None),
                _emotion=SimpleNamespace(valence=0.61),
                _cfg=SimpleNamespace(memory=MemoryConfig()),
            ))
            loop._wm.add(WMItem(
                kind="user_message",
                content="[用户消息] 记住，我叫bat，以后叫我bat。",
                priority=0.95,
            ))
            loop._wm.add(WMItem(
                kind="self_awareness",
                content="[自我感知] 连续重复查看同一路径没有新证据，应该切换策略。",
                priority=0.88,
            ))

            await CognitionLoop._consolidate(loop, active_task)

            name, found = await store.get_fact("user:name")
            assert found is True
            assert name == "bat"
            explicit = await store.list_facts(prefix="user:explicit:", limit=5)
            assert explicit

            semantic_hits = loop._semantic.retrieve("切换策略 新证据", top_k=5)
            assert any(hit["kind"] == "self_model_signal" for hit in semantic_hits)

            week_key = datetime.now(UTC).strftime("%G-W%V")
            weekly_summary = loop._semantic.get(f"daily-summary-{week_key}")
            assert weekly_summary is not None
            assert weekly_summary.kind == "daily_summary"
            assert "切换策略" in weekly_summary.body or "我叫bat" in weekly_summary.body

            narrative = loop._episodic.load_for_context(str(task_id), n_recent=2000)
            assert "切换策略" in narrative
        finally:
            await store.close()


def test_post_tick_memory_formats_learned_insight_title_with_hash_suffix():
    asyncio.run(_post_tick_memory_formats_learned_insight_title_with_hash_suffix())


async def _post_tick_memory_formats_learned_insight_title_with_hash_suffix():
    import hashlib

    from core.loop.tick import _post_tick_memory_impl
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    prefix = "当检测到行为死循环时，仅改变工具调用参数是不够的，必须在 next_step 中注入具体的、可执行的子目标内容，以强制认"
    reflection_a = prefix + "知路径A"
    reflection_b = prefix + "知路径B"

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("反思任务", goal="write insight", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            semantic = SemanticMemory(root / "semantic")
            loop = cast("Any", SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, n_recent=40000: "", record=lambda **kwargs: None),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=99),
                    emotion=SimpleNamespace(),
                ),
            ))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(
                loop,
                cast("Any", SimpleNamespace(chosen_action_id="", params={}, reflection=reflection_a, rationale="")),
                result,
                active_task,
                cycle=1,
                user_message="",
            )
            await _post_tick_memory_impl(
                loop,
                cast("Any", SimpleNamespace(chosen_action_id="", params={}, reflection=reflection_b, rationale="")),
                result,
                active_task,
                cycle=2,
                user_message="",
            )

            node_a = semantic.get(f"insight_{hashlib.md5(reflection_a.encode()).hexdigest()[:10]}")
            node_b = semantic.get(f"insight_{hashlib.md5(reflection_b.encode()).hexdigest()[:10]}")
            assert node_a is not None
            assert node_b is not None
            assert node_a.title != node_b.title
            assert node_a.title.endswith(f" [{hashlib.md5(reflection_a.encode()).hexdigest()[:6]}]")
            assert node_b.title.endswith(f" [{hashlib.md5(reflection_b.encode()).hexdigest()[:6]}]")
        finally:
            await store.close()


def test_post_tick_memory_crystallizes_event_title_with_task_id():
    asyncio.run(_post_tick_memory_crystallizes_event_title_with_task_id())


def test_post_tick_memory_skips_low_value_reflection_semantic_nodes():
    asyncio.run(_post_tick_memory_skips_low_value_reflection_semantic_nodes())


async def _post_tick_memory_crystallizes_event_title_with_task_id():
    from core.loop.tick import _post_tick_memory_impl
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("事件任务", goal="write event", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            semantic = SemanticMemory(root / "semantic")
            loop = cast("Any", SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, n_recent=40000: "", record=lambda **kwargs: None),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=1),
                    emotion=SimpleNamespace(),
                ),
            ))
            action = cast("Any", SimpleNamespace(chosen_action_id="", params={}, reflection="事件反思", rationale=""))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(loop, action, result, active_task, cycle=1, user_message="")

            ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
            node = semantic.get(f"event-task{task_id}-{ts_label}")
            assert node is not None
            assert node.title == f"[{ts_label}] task#{task_id} 事件任务"
        finally:
            await store.close()


async def _post_tick_memory_skips_low_value_reflection_semantic_nodes():
    import hashlib

    from core.loop.tick import _post_tick_memory_impl
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    reflection = "继续分析近期失败模式，下一步沉淀失败经验并观察是否改善。"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("低价值反思任务", goal="skip low value reflection", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            semantic = SemanticMemory(root / "semantic")
            loop = cast("Any", SimpleNamespace(
                _task_store=store,
                _episodic=SimpleNamespace(load_for_context=lambda task_id_str, n_recent=40000: "", record=lambda **kwargs: None),
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=1),
                    emotion=SimpleNamespace(),
                ),
            ))
            action = cast("Any", SimpleNamespace(
                chosen_action_id="",
                params={},
                reflection=reflection,
                rationale="",
                reply_to_user="我会继续处理。",
            ))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)

            await _post_tick_memory_impl(
                loop,
                action,
                result,
                active_task,
                cycle=1,
                user_message="继续",
                chat_id="chat-low-value",
            )

            insight_id = f"insight_{hashlib.md5(reflection.encode()).hexdigest()[:10]}"
            ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
            chat_digest = hashlib.md5(b"chat-low-value").hexdigest()[:12]
            chat_node = semantic.get(f"chat-summary-{chat_digest}-{ts_label}")
            turns, found = await store.get_fact(f"task:{task_id}:reflection_turns")
            assert semantic.get(insight_id) is None
            assert semantic.get(f"event-task{task_id}-{ts_label}") is None
            assert chat_node is not None
            assert reflection not in chat_node.body
            assert found is True
            assert turns == "1"
        finally:
            await store.close()


def test_post_tick_memory_crystallizes_chat_summary_and_records_chat_turns():
    asyncio.run(_post_tick_memory_crystallizes_chat_summary_and_records_chat_turns())


async def _post_tick_memory_crystallizes_chat_summary_and_records_chat_turns():
    import hashlib

    from core.loop.tick import _post_tick_memory_impl
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            task_id = await store.add_task("chat 任务", goal="write chat summary", status="in_progress")
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None

            episodic = EpisodicMemory(root / "memory")
            semantic = SemanticMemory(root / "semantic")
            loop = cast("Any", SimpleNamespace(
                _task_store=store,
                _episodic=episodic,
                _semantic=semantic,
                _emotion=SimpleNamespace(valence=0.66, arousal=0.44),
                _wm=SimpleNamespace(add=lambda item: None),
                _cfg=SimpleNamespace(
                    thresholds=SimpleNamespace(wm_pri_insight=0.8),
                    memory=SimpleNamespace(chat_crystallize_every=1),
                    emotion=SimpleNamespace(),
                ),
            ))
            action = cast("Any", SimpleNamespace(
                chosen_action_id="",
                params={},
                reflection="用户想延续这个 chat 里的远程部署排查。",
                rationale="先记录 chat 维度记忆。",
                reply_to_user="好的，我继续沿着这个 chat 的部署问题往下查。",
            ))
            result = SimpleNamespace(summary="", skipped=False, error=None, kind="execute_result", priority=0.5)
            await store.set_fact("chat:wechat:chat-1:interlocutor_profile_id", "interlocutor-bat", scope="profile")

            await _post_tick_memory_impl(
                loop,
                action,
                result,
                active_task,
                cycle=1,
                user_message="继续排查刚才这个部署问题",
                chat_id="wechat:chat-1",
            )

            ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
            digest = hashlib.md5(b"wechat:chat-1").hexdigest()[:12]
            node = semantic.get(f"chat-summary-{digest}-{ts_label}")
            assert node is not None
            assert node.kind == "chat_summary"
            assert node.source == "chat_summary"
            assert "chat:wechat:chat-1" in node.tags
            assert "继续排查刚才这个部署问题" in node.body

            chat_text = episodic.load_for_chat_context("wechat:chat-1", n_recent=20)
            assert "继续排查刚才这个部署问题" in chat_text
            assert "好的，我继续沿着这个 chat 的部署问题往下查。" in chat_text
            assert "先记录 chat 维度记忆。" not in chat_text

            interlocutor_text = episodic.load_for_interlocutor_context("interlocutor-bat")
            assert "继续排查刚才这个部署问题" in interlocutor_text
            assert "好的，我继续沿着这个 chat 的部署问题往下查。" in interlocutor_text

            turns = episodic.get_recent_turns(limit=5, chat_id="wechat:chat-1")
            assert [turn["content"] for turn in turns] == [
                "继续排查刚才这个部署问题",
                "好的，我继续沿着这个 chat 的部署问题往下查。",
            ]

            interlocutor_turns = episodic.get_recent_turns(limit=5, interlocutor_id="interlocutor-bat")
            assert [turn["content"] for turn in interlocutor_turns] == [
                "继续排查刚才这个部署问题",
                "好的，我继续沿着这个 chat 的部署问题往下查。",
            ]
        finally:
            await store.close()


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
            "repair": "bailian/qwen3.6-plus",
        },
    }

    changed = _sync_routing_models_on_primary_switch(
        cfg_data,
        old_model="copilot/gpt-5.4-mini",
        new_model="copilot/gpt-5.4-mini",
    )

    assert changed == ["reasoner"]
    assert cfg_data["routing"]["reader"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/gpt-5.4-mini"
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


def test_dev_model_target_selection_updates_reasoner_without_touching_primary_model():
    from cli.dev import _apply_model_target_selection

    cfg_data = {
        "model": "bailian/qwen3.6-plus",
        "routing": {
            "reader": "bailian/qwen-plus",
            "reasoner": "copilot/gpt-5.4",
        },
    }

    result = _apply_model_target_selection(
        cfg_data,
        current_model="bailian/qwen3.6-plus",
        new_model="copilot/o3",
        target="reasoner",
    )

    assert result["target"] == "reasoner"
    assert result["previous"] == "copilot/gpt-5.4"
    assert result["routing_changed"] == ["reasoner"]
    assert result["runtime_override_tier"] == "reasoner"
    assert cfg_data["model"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"]["reasoner"] == "copilot/o3"


def test_dev_model_target_selection_updates_vision_model_without_touching_routing():
    from cli.dev import _apply_model_target_selection

    cfg_data = {
        "model": "bailian/qwen3.6-plus",
        "vision_model": "bailian/qwen3.6-plus",
        "routing": {
            "reasoner": "copilot/gpt-5.4",
        },
    }

    result = _apply_model_target_selection(
        cfg_data,
        current_model="bailian/qwen3.6-plus",
        new_model="copilot/gpt-5.4",
        target="vision",
    )

    assert result["target"] == "vision"
    assert result["previous"] == "bailian/qwen3.6-plus"
    assert result["routing_changed"] == ["vision_model"]
    assert result["runtime_override_tier"] is None
    assert cfg_data["model"] == "bailian/qwen3.6-plus"
    assert cfg_data["routing"] == {"reasoner": "copilot/gpt-5.4"}
    assert cfg_data["vision_model"] == "copilot/gpt-5.4"


def test_dev_model_target_selection_rejects_unknown_target():
    from cli.dev import _apply_model_target_selection

    cfg_data = {
        "model": "bailian/qwen3.6-plus",
        "routing": {},
    }

    with pytest.raises(ValueError, match="未知模型目标"):
        _apply_model_target_selection(
            cfg_data,
            current_model="bailian/qwen3.6-plus",
            new_model="copilot/gpt-5.4-mini",
            target="complex",
        )


def test_dev_model_target_selection_rejects_numeric_model_id():
    from cli.dev import _apply_model_target_selection

    cfg_data = {
        "model": "bailian/qwen3.6-plus",
        "routing": {},
    }

    with pytest.raises(ValueError, match="模型 ID 不能是编号"):
        _apply_model_target_selection(
            cfg_data,
            current_model="bailian/qwen3.6-plus",
            new_model="openai-codex/4",
            target="reasoner",
        )


def test_merge_runtime_routing_override_keeps_supported_tiers_only():
    from cli.dev import _merge_runtime_routing_override

    merged = _merge_runtime_routing_override(
        {"reader": "bailian/qwen-plus", "custom": "ignored/model"},
        tier="reasoner",
        model_ref="copilot/gpt-5.4",
    )

    assert merged == {
        "reader": "bailian/qwen-plus",
        "reasoner": "copilot/gpt-5.4",
    }


def test_chat_reply_is_persisted_before_post_tick_cleanup():
    asyncio.run(_chat_reply_is_persisted_before_post_tick_cleanup())


async def _chat_reply_is_persisted_before_post_tick_cleanup():
    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
    os.environ.setdefault("GITHUB_TOKEN", "test-token")
    from core.config import Config
    from core.loop import CognitionLoop

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
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
                return cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                "Any",
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

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
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
                return cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                "Any",
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
    from core.loop.cycle.chat import _resolve_reply_chat_id
    from store.task import TaskStore

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

    cfg = Config.load(_proj_root() / "lingzhou.json.example")
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
                return cast("Any", SimpleNamespace(prediction_error=0.0, workspace_dirty=False))

            loop._perception.sense = _sense
            loop._perception.derive_cognitive_signals = lambda *args, **kwargs: cast(
                "Any",
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
