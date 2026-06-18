"""tests/test_execution_routing.py — execution 路由装配单元测试."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from core.execution import RUN_TYPE_DEFAULT_TIER, RUN_TYPE_JUDGE, RUN_TYPE_PROBE, RUN_TYPE_TOOL_CHAIN


def test_resolve_run_type_routing_forwards_cfg_overrides_catalog():
    from core.execution.routing import resolve_run_type_routing

    cfg = SimpleNamespace(run_type_routing={RUN_TYPE_JUDGE: "reader", "custom_type": "repair"})
    with patch(
        "provider.catalog.get_run_type_routing",
        return_value={RUN_TYPE_JUDGE: "reasoner", "tool_chain": "task_default"},
    ) as catalog_mock:
        routing = resolve_run_type_routing(cfg)

    catalog_mock.assert_called_once_with()
    assert routing[RUN_TYPE_JUDGE] == "reader"
    assert routing["custom_type"] == "repair"
    assert routing[RUN_TYPE_TOOL_CHAIN] == "task_default"


def test_resolve_run_type_routing_normalizes_space_in_override_keys_and_values():
    from core.execution.routing import resolve_run_type_routing

    cfg = SimpleNamespace(run_type_routing={"  judge ": " reasoner ", "\tcustom_type\n": "repair"})
    with patch("provider.catalog.get_run_type_routing", return_value={}):
        routing = resolve_run_type_routing(cfg)

    assert routing[RUN_TYPE_JUDGE] == "reasoner"
    assert routing["custom_type"] == "repair"
    assert routing[RUN_TYPE_TOOL_CHAIN] == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_TOOL_CHAIN]


def test_resolve_run_type_routing_supports_missing_cfg():
    from core.execution.routing import resolve_run_type_routing

    with patch("provider.catalog.get_run_type_routing", return_value={RUN_TYPE_JUDGE: "reader"}) as catalog_mock:
        routing = resolve_run_type_routing(None)

    catalog_mock.assert_called_once_with()
    assert routing[RUN_TYPE_JUDGE] == "reader"
    assert routing[RUN_TYPE_TOOL_CHAIN] == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_TOOL_CHAIN]


def test_resolve_run_type_routing_accepts_dict_input():
    from core.execution.routing import resolve_run_type_routing

    cfg = {
        "run_type_routing": {
            "  judge ": " reader ",
            "  custom_type  ": "repair",
        },
    }
    with patch("provider.catalog.get_run_type_routing", return_value={RUN_TYPE_TOOL_CHAIN: "task_default"}):
        routing = resolve_run_type_routing(cfg)

    assert routing[RUN_TYPE_JUDGE] == "reader"
    assert routing[RUN_TYPE_TOOL_CHAIN] == "task_default"
    assert routing["custom_type"] == "repair"


def test_resolve_run_type_routing_normalizes_case():
    from core.execution.routing import resolve_run_type_routing

    cfg = {
        "run_type_routing": {
            "  JUDGE  ": " Reader ",
            "  EXEC  ": "REASONER",
            "Tool_Chain": "Task_Default",
        },
    }
    with patch("provider.catalog.get_run_type_routing", return_value={"PROBE": "reader"}):
        routing = resolve_run_type_routing(cfg)

    assert routing["judge"] == "reader"
    assert routing["exec"] == "reasoner"
    assert routing["tool_chain"] == "task_default"


def test_resolve_run_type_routing_sanitizes_non_dict_override():
    from core.execution.routing import resolve_run_type_routing

    cfg = SimpleNamespace(run_type_routing=["not", "a", "dict"])

    with patch("provider.catalog.get_run_type_routing", return_value={RUN_TYPE_JUDGE: "reader"}):
        routing = resolve_run_type_routing(cfg)

    assert routing[RUN_TYPE_JUDGE] == "reader"
    assert routing[RUN_TYPE_TOOL_CHAIN] == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_TOOL_CHAIN]


def test_resolve_run_type_routing_falls_back_when_catalog_failures():
    from core.execution.routing import resolve_run_type_routing

    cfg = SimpleNamespace(run_type_routing={RUN_TYPE_JUDGE: "reasoner"})
    with patch("provider.catalog.get_run_type_routing", side_effect=RuntimeError("catalog read failed")):
        routing = resolve_run_type_routing(cfg)

    assert routing[RUN_TYPE_JUDGE] == "reasoner"
    assert routing[RUN_TYPE_TOOL_CHAIN] == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_TOOL_CHAIN]


def test_resolve_run_type_routing_rejects_empty_strings_after_normalize():
    from core.execution.routing import resolve_run_type_routing

    cfg = SimpleNamespace(run_type_routing={"  ": "reasoner", "judge": "   ", "": "reader", "probe": "repair"})
    with patch("provider.catalog.get_run_type_routing", return_value={}):
        routing = resolve_run_type_routing(cfg)

    assert "" not in routing
    assert routing[RUN_TYPE_JUDGE] == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_JUDGE]
    assert routing[RUN_TYPE_PROBE] == "repair"


def test_resolve_run_type_routing_rejects_non_mapping_catalog_payload():
    from core.execution.routing import resolve_run_type_routing

    cfg = SimpleNamespace(run_type_routing={RUN_TYPE_JUDGE: "reader"})
    with patch("provider.catalog.get_run_type_routing", return_value=["unexpected", "value"]):
        routing = resolve_run_type_routing(cfg)

    assert routing[RUN_TYPE_JUDGE] == "reader"
