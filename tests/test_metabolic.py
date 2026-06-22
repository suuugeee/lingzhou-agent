"""代谢引擎、免疫策略、情感状态、ethos 推导的窄测。"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# MetabolicEngine
# ══════════════════════════════════════════════════════════════════════════════


def test_state_writer_create_task_returns_task_ledger_key():
    asyncio.run(_state_writer_create_task_returns_task_ledger_key())


async def _state_writer_create_task_returns_task_ledger_key():
    from core.metabolic.proposal import StateProposal
    from core.metabolic.state_writer import apply_state_write
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "state-writer.db")
        await store.open()
        try:
            result = await apply_state_write(
                store,
                StateProposal(
                    op="create_task",
                    key="task:new",
                    value={"title": "落地器创建任务", "goal": "验证 ledger key"},
                    source="test/state_writer",
                ),
                accepted=True,
            )

            assert result.result == 1
            assert result.ledger_key == "task:1"
            assert result.accepted is True
            task = await store.get_task_by_id(1)
            assert task is not None
            assert task.title == "落地器创建任务"
        finally:
            await store.close()


def test_state_writer_unknown_op_returns_rejected_result():
    asyncio.run(_state_writer_unknown_op_returns_rejected_result())


async def _state_writer_unknown_op_returns_rejected_result():
    from core.metabolic.proposal import StateProposal
    from core.metabolic.state_writer import apply_state_write

    class _EmptyStore:
        pass

    result = await apply_state_write(
        _EmptyStore(),  # type: ignore[arg-type]
        StateProposal(op="unknown_op", key="demo", value="", source="test/state_writer"),
        accepted=True,
    )

    assert result.result is None
    assert result.ledger_key == "demo"
    assert result.accepted is False
    assert result.reason == "unknown_op"


def test_state_writer_rejects_low_value_semantic_from_internal_sources():
    asyncio.run(_state_writer_rejects_low_value_semantic_from_internal_sources())


async def _state_writer_rejects_low_value_semantic_from_internal_sources():
    from core.metabolic.proposal import StateProposal
    from core.metabolic.state_writer import apply_state_write
    from store.semantic import SemanticMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        semantic = SemanticMemory(root / "memory")
        await store.open()
        try:
            result = await apply_state_write(
                store,
                StateProposal(
                    op="add_semantic_memory",
                    key="insight-low",
                    value={
                        "id": "insight-low",
                        "kind": "learned_insight",
                        "title": "继续观察",
                        "body": "下一步继续分析近期失败模式，后续观察是否还有类似现象。",
                        "activation": 0.9,
                        "source": "loop/tick/reflection",
                    },
                    source="loop/tick/reflection",
                ),
                accepted=True,
                semantic_memory=semantic,
            )

            assert result.accepted is False
            assert result.reason == "semantic_low_value_process_note"
            assert semantic.get("insight-low") is None
        finally:
            await store.close()


def test_metabolic_submit_set_fact_writes_to_store_and_ledger():
    asyncio.run(_metabolic_submit_set_fact_writes_to_store_and_ledger())


async def _metabolic_submit_set_fact_writes_to_store_and_ledger():
    from core.metabolic import MetabolicEngine
    from core.metabolic.proposal import StateProposal
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            engine = MetabolicEngine(store)
            proposal = StateProposal(
                op="set_fact",
                key="test:key",
                value="hello",
                scope="system",
                source="test",
            )
            await engine.submit(proposal)

            val, found = await store.get_fact("test:key")
            assert found, "fact 应已写入"
            assert val == "hello"

            rows = await store.ledger_recent(limit=5)
            assert rows, "账本应有记录"
            last = rows[0]
            assert last["key"] == "test:key"
            assert last["accepted"] is True
            assert last["reason"] == ""
            assert len(last["proposal_hash"]) == 64
        finally:
            await store.close()


def test_metabolic_submit_blocked_key_skips_write_but_records_ledger():
    asyncio.run(_metabolic_submit_blocked_key_skips_write_but_records_ledger())


async def _metabolic_submit_blocked_key_skips_write_but_records_ledger():
    """evolution.evolve 被免疫器官拒绝：不写 fact，但账本仍记 accepted=False。"""
    from core.metabolic import MetabolicEngine
    from core.metabolic.proposal import StateProposal
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            engine = MetabolicEngine(store)
            # op 不是 set_fact 且在黑名单中时触发拒绝路径
            proposal = StateProposal(
                op="evolution.evolve",
                key="evolve",
                value="{}",
                scope="system",
                source="test",
            )
            await engine.submit(proposal)

            # 写入不应发生（key "evolve" 没有通过 set_fact 写入）
            val, found = await store.get_fact("evolve")
            assert not found, "被阻断的提案不应写入 fact"

            rows = await store.ledger_recent(limit=5)
            assert rows, "账本应有记录（即使被拒绝）"
            last = rows[0]
            assert last["accepted"] is False, "账本应记录 accepted=False"
            assert last["reason"], "免疫拒绝应记录原因"
            assert len(last["proposal_hash"]) == 64
        finally:
            await store.close()


def test_metabolic_submit_unknown_op_does_not_crash():
    asyncio.run(_metabolic_submit_unknown_op_does_not_crash())


async def _metabolic_submit_unknown_op_does_not_crash():
    from core.metabolic import MetabolicEngine
    from core.metabolic.proposal import StateProposal
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            engine = MetabolicEngine(store)
            proposal = StateProposal(
                op="unknown_op",
                key="unknown",
                value={"title": "test"},
                scope="system",
                source="test",
            )
            await engine.submit(proposal)

            rows = await store.ledger_recent(limit=5)
            assert rows
            assert rows[0]["op"] == "unknown_op"
            assert rows[0]["key"] == "unknown"
            assert rows[0]["accepted"] is False
            assert rows[0]["reason"] == "unknown_op"
            assert len(rows[0]["proposal_hash"]) == 64
        finally:
            await store.close()


def test_soul_change_lifecycle_writes_soul_fact_and_ledger():
    asyncio.run(_soul_change_lifecycle_writes_soul_fact_and_ledger())


async def _soul_change_lifecycle_writes_soul_fact_and_ledger() -> None:
    from core.metabolic import set_soul_fact
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            ok = await set_soul_fact(
                store,
                key="soul:test_metric",
                value='{"a": 1}',
                scope="system",
                source="test/soul_change",
                decision_basis="unit test",
            )
            assert ok is True

            raw, found = await store.get_fact("soul:test_metric")
            assert found
            assert raw == '{"a": 1}'

            rows = await store.ledger_recent(limit=5)
            assert rows
            last = rows[0]
            assert last["op"] == "soul_change"
            assert last["key"] == "soul:test_metric"
            assert last["accepted"] is True
            assert last["source"] == "test/soul_change"
        finally:
            await store.close()


def test_soul_change_rejects_non_soul_key():
    asyncio.run(_soul_change_rejects_non_soul_key())


async def _soul_change_rejects_non_soul_key() -> None:
    from core.metabolic import set_soul_fact
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            with pytest.raises(ValueError):
                await set_soul_fact(
                    store,
                    key="metric:test",
                    value="bad",
                    source="test/soul_change",
                )
        finally:
            await store.close()


def test_run_lifecycle_add_run_through_metabolic():
    asyncio.run(_run_lifecycle_add_run_through_metabolic())


async def _run_lifecycle_add_run_through_metabolic() -> None:
    from core.metabolic import add_run
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            run_id = await add_run(
                store,
                task_id=3,
                run_type="judge",
                worker_type="reader-worker",
                status="running",
                input_json={"decision": "act"},
                tool_name="demo.tool",
                source="test/run_lifecycle/add",
                decision_basis="unit_test",
            )
            assert run_id > 0

            run = await store.get_run_by_id(run_id)
            assert run is not None
            assert run.task_id == 3
            assert run.run_type == "judge"
            assert run.worker_type == "reader-worker"
            assert run.status == "running"

            rows = await store.ledger_recent(limit=5)
            assert rows
            assert rows[0]["op"] == "add_run"
            assert rows[0]["key"].startswith("run:")
            assert rows[0]["key"].endswith(str(run_id))
            assert rows[0]["source"] == "test/run_lifecycle/add"
            assert rows[0]["accepted"] is True
        finally:
            await store.close()


def test_run_lifecycle_update_run_through_metabolic():
    asyncio.run(_run_lifecycle_update_run_through_metabolic())


async def _run_lifecycle_update_run_through_metabolic() -> None:
    from core.metabolic import add_run, update_run
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            run_id = await add_run(
                store,
                task_id=7,
                run_type="tool_chain",
                worker_type="tool-chain-worker",
                status="running",
                source="test/run_lifecycle/update_before",
            )
            await update_run(
                store,
                run_id,
                status="succeeded",
                output_json={"ok": True},
                log_text="done",
                progress="100%",
                source="test/run_lifecycle/update",
                proposal_run_id=run_id,
                decision_basis="unit_test",
            )

            run = await store.get_run_by_id(run_id)
            assert run is not None
            assert run.status == "succeeded"
            assert run.output_json == {"ok": True}
            assert run.log_text == "done"
            assert run.progress == "100%"
            assert run.completed_at

            rows = await store.ledger_recent(limit=5)
            assert len(rows) >= 2
            assert any(row["op"] == "update_run" and row["key"] == f"run:{run_id}" for row in rows)
            assert any(row["op"] == "update_run" and row["source"] == "test/run_lifecycle/update" for row in rows)
            assert rows[0]["accepted"] is True
        finally:
            await store.close()


def test_run_lifecycle_compacts_large_persistent_payloads():
    asyncio.run(_run_lifecycle_compacts_large_persistent_payloads())


async def _run_lifecycle_compacts_large_persistent_payloads() -> None:
    from core.metabolic import add_run, update_run
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            huge = "A" * 20000 + "TAIL"
            items = [{"index": idx, "value": huge} for idx in range(90)]
            run_id = await add_run(
                store,
                task_id=7,
                run_type="tool_chain",
                worker_type="tool-chain-worker",
                status="running",
                source="test/run_lifecycle/compact_before",
            )
            await update_run(
                store,
                run_id,
                status="succeeded",
                output_json={"summary": huge, "items": items, "state_delta": {"body": huge}},
                log_text=huge,
                progress=huge,
                source="test/run_lifecycle/compact",
                proposal_run_id=run_id,
                decision_basis="unit_test",
            )

            run = await store.get_run_by_id(run_id)
            assert run is not None
            assert len(run.output_json["summary"]) < len(huge)
            assert "persistent storage truncated" in run.output_json["summary"]
            assert "TAIL" in run.output_json["summary"]
            assert run.output_json["items"][39]["_persistent_omitted_items"] == 11
            assert run.output_json["items"][-1]["index"] == 89
            assert len(run.log_text) < len(huge)
            assert len(run.progress) < len(huge)

            rows = await store.ledger_recent(limit=5)
            update_rows = [row for row in rows if row["op"] == "update_run"]
            assert update_rows
            assert len(update_rows[0]["value"]) <= 16000
            assert "life_ledger value truncated" in update_rows[0]["value"]
            assert "A" * 20000 not in update_rows[0]["value"]
        finally:
            await store.close()


def test_metabolic_submit_failed_write_records_ledger_and_reraises():
    asyncio.run(_metabolic_submit_failed_write_records_ledger_and_reraises())


async def _metabolic_submit_failed_write_records_ledger_and_reraises():
    from core.metabolic import MetabolicEngine
    from core.metabolic.proposal import StateProposal

    class _FailingStore:
        def __init__(self) -> None:
            self.ledger_rows: list[dict[str, Any]] = []

        async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
            raise RuntimeError(f"write failed task={task_id}")

        async def ledger_append(
            self,
            op: str,
            key: str,
            value: str,
            *,
            scope: str = "task",
            source: str = "",
            accepted: bool = True,
            run_id: int = 0,
            reason: str = "",
            proposal_hash: str = "",
            decision_basis: str = "",
        ) -> None:
            self.ledger_rows.append(
                {
                    "op": op,
                    "key": key,
                    "value": value,
                    "scope": scope,
                    "source": source,
                    "accepted": accepted,
                    "run_id": run_id,
                    "reason": reason,
                    "proposal_hash": proposal_hash,
                    "decision_basis": decision_basis,
                }
            )

    store = _FailingStore()
    engine = MetabolicEngine(store)  # type: ignore[arg-type]
    proposal = StateProposal(
        op="update_task_data",
        key="7",
        value={"inbox_messages": ["hi"]},
        scope="task",
        source="test/failure",
        run_id=42,
    )

    try:
        await engine.submit(proposal)
    except RuntimeError as exc:
        assert "write failed task=7" in str(exc)
    else:
        raise AssertionError("底层写入失败必须重新抛出")

    assert store.ledger_rows == [
        {
            "op": "update_task_data",
            "key": "7",
            "value": '{"inbox_messages": ["hi"]}',
            "scope": "task",
            "source": "test/failure",
            "accepted": False,
            "run_id": 42,
            "reason": "write_error | RuntimeError | write failed task=7",
            "proposal_hash": store.ledger_rows[0]["proposal_hash"],
            "decision_basis": "",
        }
    ]
    assert len(store.ledger_rows[0]["proposal_hash"]) == 64


def test_metabolic_submit_delete_fact_removes_store_value_and_records_ledger():
    asyncio.run(_metabolic_submit_delete_fact_removes_store_value_and_records_ledger())


async def _metabolic_submit_delete_fact_removes_store_value_and_records_ledger():
    from core.metabolic import MetabolicEngine
    from core.metabolic.proposal import StateProposal
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            await store.set_fact("focus:current_task_id", "7", scope="system")
            engine = MetabolicEngine(store)
            await engine.submit(StateProposal(
                op="delete_fact",
                key="focus:current_task_id",
                value="",
                scope="system",
                source="test/delete",
            ))

            _, found = await store.get_fact("focus:current_task_id")
            assert not found
            rows = await store.ledger_recent(limit=5)
            assert rows
            top = rows[0]
            assert top["op"] == "delete_fact"
            assert top["key"] == "focus:current_task_id"
            assert top["source"] == "test/delete"
            assert top["accepted"] is True
        finally:
            await store.close()


def test_evolution_breaker_fact_goes_through_metabolic_ledger():
    asyncio.run(_evolution_breaker_fact_goes_through_metabolic_ledger())


async def _evolution_breaker_fact_goes_through_metabolic_ledger():
    from types import SimpleNamespace

    from core.evolution.breaker import _update_target_breaker_state
    from core.metabolic import MetabolicEngine
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            engine = SimpleNamespace(
                _breaker_fail_threshold=1,
                _breaker_escalate_threshold=9,
                _breaker_cooldown_seconds=60,
                _breaker_global_cooldown_seconds=600,
            )
            ctx = SimpleNamespace(task_store=store, metabolic=MetabolicEngine(store))

            await _update_target_breaker_state(
                engine,
                ctx,
                target="demo.tool",
                success=False,
                reason="regression",
            )

            raw, found = await store.get_fact("evolution:breaker:demo.tool")
            assert found
            assert "regression" in raw
            rows = await store.ledger_recent(limit=3)
            assert rows
            top = rows[0]
            assert top["op"] == "set_fact"
            assert top["key"] == "evolution:breaker:demo.tool"
            assert top["source"] == "evolution/breaker"
            assert top["accepted"] is True
        finally:
            await store.close()


# ══════════════════════════════════════════════════════════════════════════════
# ImmunePolicy
# ══════════════════════════════════════════════════════════════════════════════


def test_immune_check_tool_blocked_allows_normal_tool():
    from core.immune.policy import check_tool_blocked

    assert check_tool_blocked("task.add") is None
    assert check_tool_blocked("memory.set_fact") is None
    assert check_tool_blocked("shell.run") is None
    assert check_tool_blocked("config.set") is None


def test_immune_check_tool_blocked_rejects_blacklisted_tools():
    from core.immune.policy import check_tool_blocked

    for tool in ("evolution.evolve", "evolution.synthesize", "soul.update",
                 "ethos.evolve", "skill.evolve", "subagent.run"):
        result = check_tool_blocked(tool)
        assert result is not None, f"{tool!r} 应被免疫阻断"
        assert "A4" in result or "黑名单" in result


def test_immune_check_tool_blocked_rejects_empty_name():
    from core.immune.policy import check_tool_blocked

    assert check_tool_blocked("") is not None


def test_immune_is_readonly_blocked_tool_blocks_mutation_tools():
    from core.immune.policy import is_readonly_blocked_tool

    # 只读子灵不能调用这些
    assert is_readonly_blocked_tool("config.set", None) is True
    assert is_readonly_blocked_tool("memory.add_semantic", None) is True
    assert is_readonly_blocked_tool("memory.set_fact", None) is True
    assert is_readonly_blocked_tool("schedule.add", None) is True
    assert is_readonly_blocked_tool("task.plan", None) is True


def test_immune_is_readonly_blocked_tool_allows_read_tools():
    from core.immune.policy import is_readonly_blocked_tool

    # 只读子灵可以使用这些
    assert is_readonly_blocked_tool("task.ask", None) is False
    assert is_readonly_blocked_tool("task.list", None) is False
    assert is_readonly_blocked_tool("memory.add_wm", None) is False
    assert is_readonly_blocked_tool("memory.drop_wm", None) is False


def test_immune_audit_evolution_target_blocks_protected_modules():
    from core.immune.policy import audit_evolution_target

    assert audit_evolution_target("core.immune.policy") is not None
    assert audit_evolution_target("core.immune.constitution") is not None
    assert audit_evolution_target("core.metabolic.engine") is not None


def test_immune_audit_evolution_target_allows_normal_modules():
    from core.immune.policy import audit_evolution_target

    assert audit_evolution_target("tools.shell") is None
    assert audit_evolution_target("core.execution") is None


# ══════════════════════════════════════════════════════════════════════════════
# EmotionState.derive_from_signals
# ══════════════════════════════════════════════════════════════════════════════


def test_emotion_high_failure_lowers_valence():
    from core.perception.emotion import EmotionState

    em = EmotionState(valence=0.65, arousal=0.50)
    em.derive_from_signals(
        failure_count=5,
        prediction_error=0.8,
        wm_pressure=0.7,
        workspace_dirty=False,
        alpha=1.0,  # 直接覆盖，无 EMA 平滑
    )
    assert em.valence < 0.65, "高失败率应降低效价"
    assert em.arousal > 0.0, "唤醒度应仍有值"


def test_emotion_low_failure_raises_valence_with_next_step():
    from core.perception.emotion import EmotionState

    em = EmotionState(valence=0.40, arousal=0.50)
    em.derive_from_signals(
        failure_count=0,
        prediction_error=0.1,
        wm_pressure=0.1,
        workspace_dirty=False,
        alpha=1.0,
        has_next_step=True,
        has_active_task=True,
        replay_trend="recovering",
    )
    assert em.valence > 0.40, "有下一步且 recovering 应提升效价"
    dominant_names = {f.name for f in em.feelings}
    assert dominant_names & {"hope", "confidence", "joy", "relief"}, \
        f"应出现正面情感，实际: {dominant_names}"


def test_emotion_recovering_trend_produces_positive_feelings():
    from core.perception.emotion import EmotionState

    em = EmotionState(valence=0.50, arousal=0.50)
    em.derive_from_signals(
        failure_count=1,
        prediction_error=0.2,
        wm_pressure=0.2,
        workspace_dirty=False,
        alpha=1.0,
        replay_trend="recovering",
    )
    names = {f.name for f in em.feelings}
    assert names & {"hope", "relief"}, f"recovering 应产生希望/宽慰，实际: {names}"


def test_emotion_derive_updates_dominant_name():
    from core.perception.emotion import EmotionState

    em = EmotionState()
    em.derive_from_signals(
        failure_count=3,
        prediction_error=0.7,
        wm_pressure=0.6,
        workspace_dirty=False,
        alpha=1.0,
    )
    assert em.dominant != "", "dominant 不应为空"


# ══════════════════════════════════════════════════════════════════════════════
# derive_ethos_state
# ══════════════════════════════════════════════════════════════════════════════


def _default_ethos_cfg() -> Any:
    from core.config_models import EthosConfig
    return EthosConfig()


def test_ethos_high_failure_triggers_prefer_verification():
    from core.perception.ethos import derive_ethos_state

    ec = _default_ethos_cfg()
    state = derive_ethos_state(
        failure_count=ec.prefer_verification_failure_count,
        high_error_streak=0,
        has_active_task=False,
        has_next_step=False,
        perception_trend="stable",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
        baseline=None,
    )
    assert state.bias.prefer_verification is True, \
        "达到 prefer_verification_failure_count 时应启用验证偏置"


def test_ethos_multiple_failures_triggers_prefer_narrow():
    from core.perception.ethos import derive_ethos_state

    ec = _default_ethos_cfg()
    state = derive_ethos_state(
        failure_count=ec.prefer_narrow_failure_count,
        high_error_streak=0,
        has_active_task=False,
        has_next_step=False,
        perception_trend="stable",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
        baseline=None,
    )
    assert state.bias.prefer_narrow_scope is True, \
        "达到 prefer_narrow_failure_count 时应收窄范围"


def test_ethos_high_error_streak_triggers_narrow():
    from core.perception.ethos import derive_ethos_state

    ec = _default_ethos_cfg()
    state = derive_ethos_state(
        failure_count=0,
        high_error_streak=ec.prefer_narrow_error_streak,
        has_active_task=False,
        has_next_step=False,
        perception_trend="stable",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
        baseline=None,
    )
    assert state.bias.prefer_narrow_scope is True, \
        "高错误 streak 应触发 prefer_narrow_scope"


def test_ethos_no_failure_keeps_baseline_values():
    from core.perception.ethos import EthosValues, derive_ethos_state

    ec = _default_ethos_cfg()
    # 提供高 baseline truth 值
    baseline = EthosValues(truth=0.90, caution=0.50, continuity=0.60,
                           curiosity=0.55, care=0.60)
    state = derive_ethos_state(
        failure_count=0,
        high_error_streak=0,
        has_active_task=False,
        has_next_step=False,
        perception_trend="stable",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
        baseline=baseline,
    )
    # EMA 混合后 truth 应接近 baseline
    assert state.values.truth >= ec.floor_truth, "truth 不应低于 floor_truth"
    # 无失败时 prefer_verification 取决于 caution 和 failure_count
    # caution=0.50 vs floor=0.05，prefer_verification_caution_min 默认 0.75
    # → 无需验证偏置（除非 failure_count 也达阈值）
    assert state.bias.prefer_verification is False, \
        "无失败且 caution 未超阈值时不应开启验证偏置"


def test_ethos_recovering_trend_boosts_curiosity():
    from core.perception.ethos import derive_ethos_state

    ec = _default_ethos_cfg()
    state_stable = derive_ethos_state(
        failure_count=0,
        high_error_streak=0,
        has_active_task=False,
        has_next_step=False,
        perception_trend="stable",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
    )
    state_recovering = derive_ethos_state(
        failure_count=0,
        high_error_streak=0,
        has_active_task=False,
        has_next_step=False,
        perception_trend="recovering",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
    )
    assert state_recovering.values.curiosity >= state_stable.values.curiosity, \
        "recovering 趋势应提升好奇心"


def test_ethos_floor_values_always_enforced():
    from core.perception.ethos import EthosValues, derive_ethos_state

    ec = _default_ethos_cfg()
    # 极低 baseline，验证 floor 兜底
    baseline = EthosValues(truth=0.0, caution=0.0, continuity=0.3,
                           curiosity=0.3, care=0.3)
    state = derive_ethos_state(
        failure_count=0,
        high_error_streak=0,
        has_active_task=False,
        has_next_step=False,
        perception_trend="stable",
        emotion_down_regulate_streak=0,
        ethos_cfg=ec,
        baseline=baseline,
    )
    assert state.values.truth >= ec.floor_truth, "truth 不应低于 floor"
    assert state.values.caution >= ec.floor_caution, "caution 不应低于 floor"


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — run_id 写入生命史账本
# ══════════════════════════════════════════════════════════════════════════════


def test_state_proposal_has_run_id_field():
    """StateProposal 应有 run_id 字段，默认为 0。"""
    from core.metabolic.proposal import StateProposal

    p = StateProposal(op="set_fact", key="x", value="v")
    assert hasattr(p, "run_id"), "StateProposal 应有 run_id 字段"
    assert p.run_id == 0, "默认 run_id 应为 0"
    p2 = StateProposal(op="set_fact", key="x", value="v", run_id=42)
    assert p2.run_id == 42


def test_ledger_append_stores_audit_fields():
    """LedgerStore.append 写入审计字段，recent() 可读回。"""
    asyncio.run(_ledger_append_stores_audit_fields())


async def _ledger_append_stores_audit_fields():
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            await store.ledger_append(
                "set_fact", "run_id_test_key", "run_id_test_val",
                scope="task",
                source="pytest",
                accepted=True,
                run_id=99,
                reason="pytest-reason",
                proposal_hash="abc123",
                decision_basis="observed user request and task state",
            )
            rows = await store.ledger_recent(limit=5)
            assert rows, "账本应有记录"
            last = rows[0]
            assert last.get("run_id") == 99, f"run_id 应为 99，实际 {last.get('run_id')!r}"
            assert last.get("reason") == "pytest-reason"
            assert last.get("proposal_hash") == "abc123"
            assert last.get("decision_basis") == "observed user request and task state"
        finally:
            await store.close()


def test_metabolic_engine_propagates_run_id_to_ledger():
    """MetabolicEngine.submit 应把 proposal.run_id 写入账本。"""
    asyncio.run(_metabolic_engine_propagates_run_id_to_ledger())


async def _metabolic_engine_propagates_run_id_to_ledger():
    import tempfile
    from pathlib import Path

    from core.metabolic import MetabolicEngine
    from core.metabolic.proposal import StateProposal
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runtime.db")
        await store.open()
        try:
            engine = MetabolicEngine(store)
            proposal = StateProposal(
                op="set_fact",
                key="engine_run_id_key",
                value="engine_run_id_val",
                scope="task",
                source="pytest",
                run_id=77,
                extras={"decision_basis": "run output changed this durable fact"},
            )
            await engine.submit(proposal)
            rows = await store.ledger_recent(limit=5)
            assert rows, "账本应有记录"
            last = rows[0]
            assert last.get("run_id") == 77, f"run_id 应为 77，实际 {last.get('run_id')!r}"
            assert last.get("reason") == ""
            assert len(last.get("proposal_hash") or "") == 64
            assert last.get("decision_basis") == "run output changed this durable fact"
        finally:
            await store.close()


def test_life_ledger_migration_adds_audit_columns():
    """旧 DB（无审计列）在 open() 后应自动迁移增加这些列。"""
    asyncio.run(_life_ledger_migration_adds_audit_columns())


async def _life_ledger_migration_adds_audit_columns():
    import tempfile
    from pathlib import Path

    import aiosqlite

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "old.db"
        # 手工建一张没有审计列的 life_ledger 表，模拟旧 DB
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS life_ledger (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT NOT NULL DEFAULT (datetime('now')),
                    op       TEXT NOT NULL,
                    key      TEXT NOT NULL,
                    value    TEXT NOT NULL DEFAULT '',
                    scope    TEXT NOT NULL DEFAULT 'task',
                    source   TEXT NOT NULL DEFAULT '',
                    accepted INTEGER NOT NULL DEFAULT 1
                )
            """)
            await db.commit()

        # 正常 open() 应触发审计列迁移
        store = TaskStore(db_path)
        await store.open()
        try:
            # 写入并读回，验证列存在且可存储
            await store.ledger_append(
                "set_fact", "migration_key", "migration_val",
                run_id=55,
                reason="migration-reason",
                proposal_hash="hash55",
                decision_basis="migration basis",
            )
            rows = await store.ledger_recent(limit=5)
            assert rows and rows[0].get("run_id") == 55, \
                f"迁移后 run_id 应可读，实际 {rows[0] if rows else '[]'!r}"
            assert rows[0].get("reason") == "migration-reason"
            assert rows[0].get("proposal_hash") == "hash55"
            assert rows[0].get("decision_basis") == "migration basis"
        finally:
            await store.close()
