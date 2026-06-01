"""core.loop.tick.exec - 执行阶段与 tick 收尾。"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

from core.metabolic import StateProposal
from store.episodic import EpisodicMemory
from tools.registry import ToolResult

from ..cycle.chat import _bind_chat_id
from ..cycle.focus import adopt_result_task, finalize_focus_task, resolve_focus_task
from ..shared.common import (
    _HINT_TIERS,
    _JUDGMENT_TIERS,
    _next_initial_tier_hint,
    _next_thinking_override,
    _should_continue_within_tick,
    _task_model_tier,
    _tool_history_entry,
)
from ..shared.continue_phase import _run_continue_phase
from ..shared.logging import (
    _clip_signal_text,
    _format_action_feedback_line,
    _summarize_state_delta,
)
from ..shared.postprocess import (
    _should_track_success_stall_tool,
    _write_success_stall_meta_reflection,
)
from ..shared.progress import (
    _action_made_progress,
    _result_fingerprint,
    action_key_param,
)
from .memory import _post_tick_memory
from .types import _build_tool_context, _log, _loop_metabolic

if TYPE_CHECKING:
    from store.task import Task


async def _execute_tick_action(
    loop: Any,
    ctx: Any,
    active_task: Any,
    action: Any,
) -> tuple[ToolResult, list[dict[str, Any]]]:
    """执行 action，并维护 behavior/tool history/读写反馈。"""
    if action.decision == "act":
        tool_id = action.chosen_action_id or ""
        key_param = action_key_param(action.params)
        current_task_id = str(active_task.id) if active_task else None
        for item in loop._behavior.on_act(tool_id, key_param, current_task_id, action.params):
            loop._wm.add(item)
    else:
        for item in loop._behavior.on_wait(action.decision, active_task is not None):
            loop._wm.add(item)
        _llm_skipped = (action.rationale or "").startswith("[按请求聚合]")
        if not _llm_skipped and loop._task_store is not None:
            _rt = "chat_reply" if action.speech_intent else "judge"
            try:
                await loop._task_store.add_run(
                    task_id=active_task.id if active_task else 0,
                    run_type=_rt,
                    worker_type=f"{_rt}-worker",
                    status="succeeded",
                    output_json={"decision": action.decision, "rationale": (action.rationale or "")},
                )
            except Exception as _exc:
                _log.debug("[tick] judge/chat_reply run 写入失败（不影响主流程）: %s", _exc)

    result = await loop._run_driver.dispatch(action, ctx)
    tool_history: list[dict[str, Any]] = []
    if action.decision == "act":
        tool_history.append(_tool_history_entry(action, result))
        loop._behavior.on_act_result(action.chosen_action_id or "", result.summary or "")

    if action.decision == "act" and not result.error:
        tool = action.chosen_action_id or ""
        path = (action.params or {}).get("path") or ""
        if tool == "file.read":
            max_chars = int((action.params or {}).get("max_chars") or 4000)
            start = int((action.params or {}).get("start") or 0)
            end = int((action.params or {}).get("end") or 0)
            for item in loop._behavior.on_read(path, max_chars, result.summary, start=start, end=end):
                loop._wm.add(item)
        elif tool == "file.list":
            for item in loop._behavior.on_list(path, result.summary):
                loop._wm.add(item)
    if action.decision == "act":
        tool = action.chosen_action_id or ""
        if tool == "file.edit" and result.error and "OldTextNotFound" in result.error:
            for item in loop._behavior.on_edit_failure(result.error):
                loop._wm.add(item)

    return result, tool_history


async def _maybe_run_tick_continue_phase(
    loop: Any,
    ctx: Any,
    user_message: str,
    active_task: Any,
    cognitive_signals: Any,
    action: Any,
    result: ToolResult,
    tool_history: list[dict[str, Any]],
) -> tuple[Any, ToolResult]:
    """按需执行同 tick 的 continue phase。"""
    if not _should_continue_within_tick(
        action,
        user_message=user_message,
        has_active_task=active_task is not None,
        registry=loop._registry,
    ):
        return action, result
    return await _run_continue_phase(
        loop=loop,
        ctx=ctx,
        user_message=user_message,
        active_task=active_task,
        cognitive_signals=cognitive_signals,
        action=action,
        result=result,
        tool_history=tool_history,
    )


async def _sync_tick_action_state(
    loop: Any,
    action: Any,
    result: ToolResult | Any,
    active_task: Task | None,
    cycle: int,
    chat_id: str | None,
) -> Task | None:
    previous_task_next_step = (active_task.next_step or "") if active_task else ""
    focus_task = await adopt_result_task(loop, active_task, action, result)
    prev_sig = loop._last_action_sig
    prev_fp = loop._last_result_fp
    cur_sig = f"{action.chosen_action_id or ''}|{action_key_param(action.params)}" if action.decision == "act" else ""
    cur_fp = _result_fingerprint(result.summary) if action.decision == "act" and not result.error and not result.skipped else ""

    loop._last_next_step = action.next_step or ""
    loop._last_decision = action.decision
    loop._last_act_progressful, loop._last_act_progress_reason = _action_made_progress(
        action,
        result,
        prev_sig=prev_sig,
        prev_fp=prev_fp,
        registry=loop._registry,
    )
    loop._last_action_tool = action.chosen_action_id or ""
    loop._last_action_key = action_key_param(action.params) if action.decision == "act" else ""
    loop._last_action_summary = _clip_signal_text(result.summary or "") if action.decision == "act" else ""
    loop._last_action_error = _clip_signal_text(result.error or "", 100) if action.decision == "act" else ""
    loop._last_action_state_delta = _summarize_state_delta(result.state_delta) if action.decision == "act" else ""

    if action.decision == "act":
        if result.error:
            loop._last_action_status = "error"
        elif result.skipped:
            loop._last_action_status = "skipped"
        else:
            loop._last_action_status = "ok"
    else:
        loop._last_action_status = action.decision

    loop._recent_action_feedback.append(
        _format_action_feedback_line(
            action,
            result,
            progressful=loop._last_act_progressful,
        )
    )
    loop._last_action_sig = cur_sig
    loop._last_result_fp = cur_fp

    focus_previous_next_step = previous_task_next_step
    if focus_task is not None and (active_task is None or focus_task.id != active_task.id):
        focus_previous_next_step = str(getattr(focus_task, "next_step", "") or "")

    from core.loop.task.runtime import _sync_task_progress_state

    active_task = await _sync_task_progress_state(
        loop._task_store,
        focus_task,
        previous_next_step=focus_previous_next_step,
        action=action,
        progressful=loop._last_act_progressful,
        state_delta=result.state_delta,
    )
    await _bind_chat_id(loop, active_task, chat_id)
    await _maybe_record_success_stall_reflection(loop, active_task, action, result, cycle)
    return active_task


async def _apply_tick_model_strategy(
    loop: Any,
    action: Any,
    active_task: Task | None,
) -> Task | None:
    cfg = loop._cfg
    next_tier = _next_initial_tier_hint(action) or ""
    task_tier = _task_model_tier(active_task)
    persist_tier = next_tier if next_tier in _JUDGMENT_TIERS else (task_tier if task_tier in _JUDGMENT_TIERS else "")

    if active_task and persist_tier and persist_tier != task_tier:
        await loop._task_store.update_task_data(active_task.id, {"model_tier": persist_tier})
        active_task.model_tier = persist_tier

    if next_tier in _JUDGMENT_TIERS:
        loop._pending_tier = next_tier
    else:
        loop._pending_tier = None

    strategy = action.model_strategy or {}
    idle_gap_ms = strategy.get("next_idle_gap_ms")
    idle_gap_secs = strategy.get("next_idle_gap_secs")
    raw_gap = (float(idle_gap_ms) / 1000.0) if idle_gap_ms is not None else (idle_gap_secs if idle_gap_secs is not None else None)
    if raw_gap is not None:
        try:
            gap_f = float(raw_gap)
            has_task = (await resolve_focus_task(loop)) is not None
            if has_task:
                bounds = cfg.loop.idle_with_task_bounds
                lo, hi = float(bounds[0]) / 1000.0, float(bounds[1]) / 1000.0
            else:
                bounds = cfg.loop.idle_no_task_bounds
                lo, hi = (float(bounds[0]) / 1000.0, float(bounds[1]) / 1000.0) if len(bounds) >= 2 else (5.0, 300.0)
            loop._pending_idle_gap = max(lo, min(hi, gap_f * (2.0 if not getattr(loop, '_last_act_progressful', True) else 1.0)))
        except (TypeError, ValueError):
            loop._pending_idle_gap = None
    else:
        loop._pending_idle_gap = None

    raw_overrides = strategy.get("routing_overrides")
    if isinstance(raw_overrides, dict):
        if not raw_overrides:
            loop._pending_routing_overrides = None
            await _loop_metabolic(loop).submit(StateProposal(
                op="set_fact", key="pref:routing_overrides", value="",
                scope="system", source="loop/tick/routing",
            ))
        else:
            valid = {
                key: value for key, value in raw_overrides.items()
                if key in _HINT_TIERS and isinstance(value, str) and value
            }
            if valid:
                loop._pending_routing_overrides = valid
                await _loop_metabolic(loop).submit(StateProposal(
                    op="set_fact", key="pref:routing_overrides", value=json.dumps(valid),
                    scope="system", source="loop/tick/routing",
                ))
            else:
                loop._pending_routing_overrides = None
                await _loop_metabolic(loop).submit(StateProposal(
                    op="set_fact", key="pref:routing_overrides", value="",
                    scope="system", source="loop/tick/routing",
                ))

    loop._pending_thinking_override = _next_thinking_override(strategy)
    return active_task


async def _persist_tick_post_state(
    loop: Any,
    action: Any,
    active_task: Task | None,
    cycle: int,
    ethos_state: Any = None,
) -> None:
    await _loop_metabolic(loop).submit(StateProposal(
        op="set_fact", key="soul:emotion_state",
        value=json.dumps({
            "valence": round(loop._emotion.valence, 4),
            "arousal": round(loop._emotion.arousal, 4),
            "dominance": round(loop._emotion.dominance, 4),
        }),
        source="loop/tick/post_state",
    ))

    if ethos_state is not None:
        await _loop_metabolic(loop).submit(StateProposal(
            op="set_fact", key="soul:ethos_baseline",
            value=json.dumps({
                "truth": ethos_state.values.truth,
                "caution": ethos_state.values.caution,
                "continuity": ethos_state.values.continuity,
                "curiosity": ethos_state.values.curiosity,
                "care": ethos_state.values.care,
            }),
            source="loop/tick/post_state",
        ))

    import datetime as _dt

    try:
        state_dir = loop._cfg.state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tick": cycle,
            "ts": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "active_task_id": str(active_task.id) if active_task else None,
            "active_task_title": active_task.title if active_task else None,
            "active_task_goal": (active_task.goal or "") if active_task else None,
            "last_decision": action.decision,
            "last_action": (
                f"{action.chosen_action_id} {action_key_param(action.params)}"
                if action.decision == "act" else action.decision
            ),
            "emotion": {
                "valence": round(loop._emotion.valence, 3),
                "arousal": round(loop._emotion.arousal, 3),
            },
            "exit_type": "crash",
        }
        _p = state_dir / "survival.json"
        _p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception as _e:
        _log.debug("[survival] 写入 survival.json 失败: %s", _e)

    for belief_item in loop._behavior.on_judgment(action.rationale or ""):
        loop._wm.add(belief_item)


async def _run_tick_maintenance(loop: Any, active_task: Task | None, cycle: int) -> None:
    cfg = loop._cfg
    wm_pressure = loop._wm.pressure
    if (
        wm_pressure < cfg.memory.consolidate_low_pressure_skip_threshold
        and (cycle % cfg.loop.consolidate_every != 0 or wm_pressure < loop._cfg.thresholds.wm_pressure_task)
    ):
        return

    await loop._consolidate(active_task)
    try:
        _gm = EpisodicMemory.narrative_path_for_dir(loop._cfg.memory_dir, None)
        if not _gm.exists():
            _gm = EpisodicMemory.legacy_narrative_path_for_dir(loop._cfg.memory_dir, None)
        if _gm.exists():
            _sz = _gm.stat().st_size
            _lc = len(_gm.read_text().splitlines())
            if _sz > cfg.memory.global_md_warn_bytes or _lc > cfg.memory.global_md_warn_lines:
                from memory.working import WMItem

                loop._wm.add(WMItem(
                    kind="self_awareness",
                    content=f"[记忆压力] global.md 当前 {_lc} 行 / {_sz} 字节。",
                    priority=0.75,
                ))
    except Exception:
        pass

    await loop._soul.sync_md()
    with contextlib.suppress(Exception):
        await loop._task_store.wal_checkpoint()


async def _maybe_run_tick_evolution(loop: Any, cycle: int, perception_replay: Any) -> None:
    cfg = loop._cfg
    if perception_replay is None:
        return
    should_evolve = (
        cfg.evolution.enabled and (
            perception_replay.high_error_streak >= cfg.evolution.error_streak_evolve
            or cycle % cfg.loop.evolve_every == 0
        )
    )
    if not should_evolve:
        return

    ctx = _build_tool_context(loop)
    results = await loop._evolution.run(ctx)
    for evolve_result in results:
        if evolve_result.success:
            from .prep import console

            console.print(f"[green][evolution] {evolve_result.target} 已进化[/green]")
            if evolve_result.target.startswith("prompt:"):
                prompt_key = evolve_result.target.split(":", 1)[1]
                loop._judgment.reload_prompt(prompt_key)
    await loop._soul.refresh_identity(loop._judgment)


async def _tick_finalize_impl(
    loop: Any,
    action: Any,
    result: ToolResult | Any,
    active_task: Task | None,
    cycle: int,
    user_message: str,
    chat_id: str | None = None,
    perception_replay: Any = None,
    ethos_state: Any = None,
) -> str:
    post_tick_memory = getattr(loop, "_post_tick_memory", None)
    if callable(post_tick_memory):
        if getattr(post_tick_memory, "__self__", None) is loop:
            await post_tick_memory(action, result, active_task, cycle, user_message, chat_id)
        else:
            await post_tick_memory(loop, action, result, active_task, cycle, user_message, chat_id)
    else:
        await _post_tick_memory(loop, action, result, active_task, cycle, user_message, chat_id)

    await _run_tick_maintenance(loop, active_task, cycle)
    await _maybe_run_tick_evolution(loop, cycle, perception_replay)

    active_task = await _sync_tick_action_state(loop, action, result, active_task, cycle, chat_id)
    active_task = await finalize_focus_task(
        loop,
        action=action,
        active_task=active_task,
        chat_id=chat_id,
        user_message=user_message,
    )
    active_task = await _apply_tick_model_strategy(loop, action, active_task)
    await _persist_tick_post_state(loop, action, active_task, cycle, ethos_state=ethos_state)

    return action.reply_to_user


async def _maybe_record_success_stall_reflection(
    loop: Any,
    active_task: Task | None,
    action: Any,
    result: ToolResult,
    cycle: int,
) -> None:
    tool_name = action.chosen_action_id or ""
    qualifies = (
        active_task is not None
        and action.decision == "act"
        and not result.error
        and not result.skipped
        and not loop._last_act_progressful
        and _should_track_success_stall_tool(tool_name, loop._registry)
    )
    if not qualifies:
        loop._success_stall_task_id = str(active_task.id) if active_task else None
        loop._success_stall_streak = 0
        return

    assert active_task is not None
    task_id = str(active_task.id)
    if loop._success_stall_task_id != task_id:
        loop._success_stall_task_id = task_id
        loop._success_stall_streak = 0

    loop._success_stall_streak += 1
    if loop._success_stall_streak != 2:
        return

    await _write_success_stall_meta_reflection(
        loop._task_store,
        active_task,
        action,
        result,
        streak=loop._success_stall_streak,
        cycle=cycle,
        metabolic=_loop_metabolic(loop),
    )
