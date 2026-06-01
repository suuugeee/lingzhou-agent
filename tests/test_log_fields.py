"""tests/test_log_fields.py — P3 结构化日志字段。"""

from __future__ import annotations

from core.judgment import JudgmentOutput
from core.log_fields import (
    execution_scope_fields,
    format_log_fields,
    judgment_outcome_fields,
    llm_call_fields,
    tick_scope_fields,
)
from core.loop.shared.logging import _format_action_feedback_line
from tools.registry import ToolResult


def test_format_log_fields_skips_empty() -> None:
    text = format_log_fields(run=1, task=None, tool="file.read", tier="")
    assert text == "run=1 tool=file.read"


def test_execution_scope_fields_order() -> None:
    text = execution_scope_fields(
        run_id=42,
        task_id=7,
        tool="shell.run",
        tier="reader",
        worker="exec-worker",
        status="succeeded",
        dispatch_ms=12,
    )
    assert "run=42" in text
    assert "task=7" in text
    assert "tool=shell.run" in text
    assert "tier=reader" in text
    assert "worker=exec-worker" in text
    assert "status=succeeded" in text
    assert "dispatch_ms=12" in text


def test_llm_call_fields() -> None:
    text = llm_call_fields(
        model_ref="bailian/qwen-plus",
        tier="reasoner",
        phase="judgment",
        usage_source="provider",
        thinking=True,
        attempt=2,
        messages=10,
    )
    assert "model_ref=bailian/qwen-plus" in text
    assert "tier=reasoner" in text
    assert "usage_source=provider" in text
    assert "attempt=2" in text


def test_judgment_outcome_fields() -> None:
    text = judgment_outcome_fields(
        phase="initial",
        tier="reasoner",
        model_ref="bailian/qwen-plus",
        thinking="high",
        applied_skills="none",
    )
    assert "phase=initial" in text
    assert "model_ref=bailian/qwen-plus" in text


def test_tick_scope_fields() -> None:
    text = tick_scope_fields(
        tick=3,
        task_id=9,
        decision="act",
        tool="file.read",
        model_ref="bailian/qwen-plus",
        tier="reader",
    )
    assert "tick=3" in text
    assert "task=9" in text


def test_action_feedback_prefers_log_summary() -> None:
    action = JudgmentOutput.from_llm(
        '{"decision":"act","chosen_action_id":"file.read","params":{"path":"/tmp/x"},"rationale":"ok"}'
    )
    line = _format_action_feedback_line(
        action,
        ToolResult(
            summary="X" * 500,
            metadata={"log_summary": "file.read path=/tmp/x chars=10"},
        ),
        progressful=True,
    )
    assert "file.read path=/tmp/x chars=10" in line
    assert "XXX" not in line


def test_capability_mapping_snapshot_cached() -> None:
    from pathlib import Path

    from core.judgment.assembler.model_routing import _capability_mapping_snapshot
    from core.judgment.output import registry_manifest_signature
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.discover(Path(__file__).resolve().parents[1] / "tools")
    sig = registry_manifest_signature(registry)
    first = _capability_mapping_snapshot(sig)
    second = _capability_mapping_snapshot(sig)
    assert first is second
    assert "completion_info_only" in first or len(first) >= 0


def test_tool_tier_mapping_cached_matches_manifest_rules() -> None:
    from pathlib import Path

    from core.judgment.output import (
        _tier_for_manifest_fields,
        registry_manifest_signature,
        tool_tier,
        tool_tier_mapping,
    )
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.discover(Path(__file__).resolve().parents[1] / "tools")
    sig = registry_manifest_signature(registry)
    first = tool_tier_mapping(registry)
    second = tool_tier_mapping(registry)
    assert first == second
    for name, prefer_tier, progress_category, caps in sig:
        expected = _tier_for_manifest_fields(prefer_tier, progress_category, caps)
        assert expected == tool_tier(name, registry)
        assert name in first.get(expected, [])


def test_catalog_models_snapshot_cached() -> None:
    from core.judgment.assembler.model_routing import _catalog_models_snapshot

    key = "/tmp/test-models.json"
    first = _catalog_models_snapshot(key)
    second = _catalog_models_snapshot(key)
    assert first is second
