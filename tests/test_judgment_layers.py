"""P1 judgment 分层与 ToolResult metadata 契约回归。"""
from __future__ import annotations

import importlib

import pytest


def test_boundary_normalize_exports() -> None:
    from core.judgment.boundary import coerce_reply_only_output, normalize_action_shape

    assert callable(normalize_action_shape)
    assert callable(coerce_reply_only_output)


def test_executor_uses_routing_and_health_mixins() -> None:
    from core.judgment.decision.health_mixin import ExecutorHealthMixin
    from core.judgment.decision.routing_mixin import ExecutorRoutingMixin
    from core.judgment.executor import JudgmentExecutor

    assert issubclass(JudgmentExecutor, ExecutorRoutingMixin)
    assert issubclass(JudgmentExecutor, ExecutorHealthMixin)


def test_tool_history_compact_limits_from_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import Config
    from core.judgment.policy import tool_history_compact_limits

    monkeypatch.setenv("TEST_API_KEY", "x")
    cfg = Config.model_validate(
        {
            "model": "testprov/m",
            "providers": {
                "testprov": {
                    "type": "openai_compat",
                    "base_url": "http://127.0.0.1/v1",
                    "api_key_env": "TEST_API_KEY",
                    "models": [{"id": "m"}],
                }
            },
            "thresholds": {
                "continue_tool_history_compact_threshold": 12,
                "continue_tool_history_keep_last": 4,
            },
        }
    )
    threshold, keep_last = tool_history_compact_limits(cfg)
    assert threshold == 12
    assert keep_last == 4


def test_tool_metadata_shape() -> None:
    from tools.registry import tool_metadata

    meta = tool_metadata("file.read", "file.read path=/tmp/a chars=10", path="/tmp/a", chars=10)
    assert meta["tool_name"] == "file.read"
    assert meta["log_summary"].startswith("file.read")
    assert meta["path"] == "/tmp/a"


@pytest.mark.asyncio
async def test_normalize_judgment_output_unknown_tool_becomes_wait() -> None:
    from core.judgment.boundary import normalize_judgment_output
    from core.judgment.output import JudgmentOutput

    class _Executor:
        async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
            return None

    output = JudgmentOutput(
        decision="act",
        chosen_action_id="not.a.real.tool",
        params={},
        rationale="",
    )
    class _Registry:
        def get(self, name: str) -> None:
            return None

    out = await normalize_judgment_output(
        _Executor(),
        output,
        context_text="",
        raw="{}",
        registry=_Registry(),
    )
    assert out.decision == "wait"
    assert "未知工具" in (out.rationale or "")


@pytest.mark.asyncio
async def test_problem_solving_guard_blocks_non_workbench_actions() -> None:
    from core.judgment.boundary import normalize_judgment_output
    from core.judgment.output import JudgmentOutput

    class _Executor:
        async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
            return None

    class _Registry:
        def get(self, name: str):
            return object()

    context = (
        "### 通用问题解决守卫\n"
        "guard=active\n"
        "signals=user_correction, workbench_incomplete\n"
        "missing_fields=domain, intent\n"
        "\n### 近期关键事实\n"
    )

    out = await normalize_judgment_output(
        _Executor(),
        JudgmentOutput(
            decision="act",
            chosen_action_id="shell.run",
            params={"command": "git push"},
            reply_to_user="我继续推送。",
            rationale="继续旧路径",
        ),
        context_text=context,
        raw="{}",
        registry=_Registry(),
    )

    assert out.decision == "wait"
    assert out.chosen_action_id == ""
    assert out.params == {}
    assert out.reply_to_user == ""
    assert "通用问题解决守卫已触发" in out.rationale
    assert "task.workbench" in out.rationale


@pytest.mark.asyncio
async def test_problem_solving_guard_allows_workbench_actions() -> None:
    from core.judgment.boundary import normalize_judgment_output
    from core.judgment.output import JudgmentOutput

    class _Executor:
        async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
            return None

    class _Registry:
        def get(self, name: str):
            return object()

    context = "### 通用问题解决守卫\nguard=active\n\n### 近期关键事实\n"

    out = await normalize_judgment_output(
        _Executor(),
        JudgmentOutput(
            decision="act",
            chosen_action_id="task.workbench",
            params={"workbench": {"domain": "git"}},
            rationale="先固化工作台",
        ),
        context_text=context,
        raw="{}",
        registry=_Registry(),
    )

    assert out.decision == "act"
    assert out.chosen_action_id == "task.workbench"
    assert out.params == {"workbench": {"domain": "git"}}


@pytest.mark.asyncio
async def test_problem_solving_guard_allows_action_first_tool_actions() -> None:
    from core.judgment.boundary import normalize_judgment_output
    from core.judgment.output import JudgmentOutput

    class _Executor:
        async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
            return None

    class _Registry:
        def get(self, name: str):
            return object()

    context = (
        "### 任务级皮层工作区\n"
        "action_first:\n"
        "- intent=execute\n"
        "- must_act=yes\n"
        "- minimum_next_action=先对用户给定的强输入做最小可验证动作\n"
        "\n### 通用问题解决守卫\n"
        "guard=active\n"
        "signals=diagnostic_or_repair_intent, action_first_required, workbench_incomplete\n"
        "missing_fields=domain, intent\n"
        "\n### 近期关键事实\n"
    )

    out = await normalize_judgment_output(
        _Executor(),
        JudgmentOutput(
            decision="act",
            chosen_action_id="shell.run",
            params={"command": "curl -L https://example.com/sub -o /tmp/sub.yaml"},
            rationale="先执行最小验证动作",
        ),
        context_text=context,
        raw="{}",
        registry=_Registry(),
    )

    assert out.decision == "act"
    assert out.chosen_action_id == "shell.run"
    assert out.params["command"].startswith("curl -L")


@pytest.mark.asyncio
async def test_action_first_wait_falls_back_to_web_fetch_for_captured_url() -> None:
    from core.judgment.boundary import normalize_judgment_output
    from core.judgment.output import JudgmentOutput

    class _Executor:
        async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
            return None

    class _Registry:
        def get(self, name: str):
            return object() if name in {"web.fetch", "task.list"} else None

    context = (
        "### 任务级皮层工作区\n"
        "action_first:\n"
        "- intent=execute\n"
        "- must_act=yes\n"
        "captured_inputs:\n"
        "- url=https://example.com/sub?clash=1\n"
        "\n### 通用问题解决守卫\n"
        "guard=active\n"
        "signals=action_first_required, workbench_incomplete\n"
        "missing_fields=domain, intent\n"
    )

    out = await normalize_judgment_output(
        _Executor(),
        JudgmentOutput(decision="wait", rationale="我下一轮处理"),
        context_text=context,
        raw="{}",
        registry=_Registry(),
    )

    assert out.decision == "act"
    assert out.chosen_action_id == "web.fetch"
    assert out.params == {"url": "https://example.com/sub?clash=1", "max_chars": 20000}
    assert "Action-first fallback" in out.rationale


def test_judgment_subpackages_importable() -> None:
    for name in ("core.judgment.boundary", "core.judgment.decision", "core.judgment.policy"):
        mod = importlib.import_module(name)
        assert mod is not None


def test_cognition_frame_from_frame_module() -> None:
    from core.judgment import CognitionFrame as exported
    from core.judgment.frame import CognitionFrame as canonical

    assert exported is canonical


def test_decision_rounds_importable() -> None:
    from core.judgment.decision.rounds import JudgmentRoundDeps, decide_initial

    assert JudgmentRoundDeps is not None
    assert callable(decide_initial)


def test_context_budget_and_signals_modules() -> None:
    from core.judgment.context.budget import apply_context_budget
    from core.judgment.context.budget import apply_context_budget as budget_fn
    from core.judgment.context.signals import _fmt_judgment_signals

    assert apply_context_budget is budget_fn
    assert callable(_fmt_judgment_signals)


def test_context_sections_module() -> None:
    from core.judgment.context.sections import _fmt_current_time, _fmt_wm
    from core.judgment.context.sections import _fmt_current_time as from_sections

    assert callable(_fmt_wm)
    assert _fmt_current_time() == from_sections()


def test_trim_messages_omits_whole_messages_not_slices() -> None:
    from types import SimpleNamespace

    from core.judgment.decision.helpers import (
        _PROMPT_OVERFLOW_OMIT_STUB,
        _trim_messages_for_prompt_limit_impl,
    )
    from core.judgment.decision.prompt_mixin import ExecutorPromptMixin
    from provider.base import Message

    long_body = "UNIQUE_BODY_MARKER_" + ("正文" * 400)
    messages = [
        Message(role="tool", content=long_body),
        Message(role="assistant", content="短回复"),
        Message(role="user", content="最新用户问题"),
    ]
    executor = SimpleNamespace(_estimate_text_tokens=ExecutorPromptMixin._estimate_text_tokens)
    trimmed = _trim_messages_for_prompt_limit_impl(executor, messages, prompt_limit=200)

    assert trimmed is not messages
    for msg in trimmed:
        content = str(getattr(msg, "content", "") or "")
        assert "UNIQUE_BODY_MARKER_" not in content or content == long_body
        assert "[prompt 已压缩]" not in content
        assert "...[prompt" not in content
    assert any(
        str(getattr(m, "content", "")) == _PROMPT_OVERFLOW_OMIT_STUB for m in trimmed
    )


def test_context_utils_module() -> None:
    from core.judgment.context.utils import (
        _clear_context_cache,
        _compress_text_segments,
        _estimate_tokens,
        _fill_template,
        _validate_context_schema,
    )
    from core.judgment.context.utils import (
        _fill_template as fill_fn,
    )
    from core.judgment.context.utils import (
        _validate_context_schema as schema_fn,
    )

    assert _fill_template is fill_fn
    assert _validate_context_schema is schema_fn
    _clear_context_cache()
    long_body = "段落一\n\n" + ("x" * 200) + "\n\n段落尾"
    compressed = _compress_text_segments(long_body, keep_tokens=_estimate_tokens("短") * 2)
    assert len(compressed) < len(long_body)
    ok, _ = _validate_context_schema(
        {"identity": {}, "tasks": {}, "memory": {}, "perception": {}}
    )
    assert ok is True


def test_routing_posture_converge_when_explore_high() -> None:
    from core.judgment.policy import routing_posture

    assert routing_posture(user_message="", task_explore_count=5, task_explore_converge_after=3) == "converge"
    assert routing_posture(user_message="hi", task_explore_count=99, task_explore_converge_after=3) == "respond"


def test_continue_phase_policy_payload_uses_shared_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import Config
    from core.judgment.policy import continue_phase_policy_payload

    monkeypatch.setenv("TEST_API_KEY", "x")
    cfg = Config.model_validate(
        {
            "model": "testprov/m",
            "providers": {
                "testprov": {
                    "type": "openai_compat",
                    "base_url": "http://127.0.0.1/v1",
                    "api_key_env": "TEST_API_KEY",
                    "models": [{"id": "m"}],
                }
            },
            "thresholds": {
                "continue_tool_history_compact_threshold": 10,
                "continue_tool_history_keep_last": 3,
            },
        }
    )
    payload = continue_phase_policy_payload(cfg, tool_history_count=10)
    assert payload["tool_history_compact_threshold"] == 10
    assert payload["tool_history_will_compact_next"] is True
