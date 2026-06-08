"""core.loop.tick.prep - 感知阶段与判断阶段准备。"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from rich.console import Console

from core.immune import extract_constitution_boundaries, load_constitution
from core.log_fields import tick_scope_fields
from core.loop.runs.refresh import refresh_running_runs
from core.loop.task.runtime import (
    _consume_task_runtime_hints,
    _ingest_actionable_meta_reflections,
)
from core.metabolic import update_task_data
from core.perception import (
    EthosValues,
    build_emotion_replay,
    build_perception_replay,
    compute_judgment_signals,
    derive_ethos_state,
)
from memory.working import WMItem

from ..cycle.chat import _bind_chat_id
from ..cycle.focus import claim_focus_task, prepare_focus_task, task_matches_chat
from ..shared.common import (
    _perception_replay_fallback,
    _prefer_tier_for_task,
    _resolve_thinking_override,
    _thinking_floor,
)
from .types import (
    _LLM_WAKE_WM_KINDS,
    _log,
    _loop_metabolic,
    _TickJudgmentPrep,
)

console = Console()


def _should_steer_active_task_from_user_message(active_task: Any, user_message: str) -> bool:
    return active_task is not None and bool(str(user_message or "").strip())


async def _maybe_steer_active_task_from_user_message(
    task_store: Any,
    active_task: Any,
    user_message: str,
    metabolic: Any | None = None,
) -> Any:
    if not _should_steer_active_task_from_user_message(active_task, user_message):
        return active_task
    message = (
        "收到新的用户消息："
        f"{str(user_message or '').strip()}"
    )
    extras = getattr(active_task, "extras", None)
    existing = extras.get("inbox_messages") if isinstance(extras, dict) else []
    if not isinstance(existing, list):
        existing = []
    if message in existing:
        return active_task
    existing = [*existing, message]
    update: dict[str, Any] = {"inbox_messages": existing}
    is_self_drive = getattr(active_task, "source", None) == "self_drive"
    if is_self_drive:
        update["had_user_inbox"] = True
    writer = metabolic if metabolic is not None else task_store
    await update_task_data(writer, active_task.id, update, source="loop/tick/user_inbox")
    active_task.extras = dict(extras) if isinstance(extras, dict) else {}
    active_task.extras["inbox_messages"] = existing
    if is_self_drive:
        active_task.extras["had_user_inbox"] = True
    _log.info(
        "[task-inbox] active_task=%s queued new user instruction into inbox",
        active_task.id,
    )
    return active_task


async def _consume_active_task_inbox(task_store: Any, active_task: Any) -> Any:
    if active_task is None:
        return None
    extras = getattr(active_task, "extras", None)
    if not isinstance(extras, dict):
        return active_task
    raw_messages = extras.get("inbox_messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return active_task
    messages = await task_store.pop_task_inbox(active_task.id)
    if messages:
        active_task.extras = dict(extras)
        active_task.extras["inbox_messages"] = messages
    return active_task


async def _prepare_active_task_for_tick(loop: Any, user_message: str, chat_id: str | None) -> Any:
    active_task = await prepare_focus_task(loop, user_message=user_message, chat_id=chat_id)
    if (
        active_task is not None
        and str(getattr(active_task, "source", "") or "") == "self_drive"
        and str(user_message or "").strip()
        and (not str(chat_id or "").strip() or not await task_matches_chat(loop, active_task, chat_id))
    ):
        _log.info(
            "[focus] ignore self_drive task=%s for new user message chat_id=%s",
            getattr(active_task, "id", ""),
            str(chat_id or "").strip() or "-",
        )
        active_task = None
    await _ingest_actionable_meta_reflections(loop._task_store, loop._wm, metabolic=_loop_metabolic(loop))
    active_task = await _consume_task_runtime_hints(loop._task_store, active_task, loop._wm, metabolic=_loop_metabolic(loop))
    active_task = await _maybe_steer_active_task_from_user_message(
        loop._task_store,
        active_task,
        user_message,
        _loop_metabolic(loop),
    )
    active_task = await _consume_active_task_inbox(loop._task_store, active_task)
    await _bind_chat_id(loop, active_task, chat_id)
    await claim_focus_task(loop, active_task, chat_id=chat_id, clear_current=not bool(str(chat_id or "").strip()))

    if not user_message:
        await loop._maybe_inject_self_drive()
        if loop._bootstrap_mode == "full":
            if active_task is not None:
                content = (
                    "[初始化未完成] BOOTSTRAP.md 仍然存在，初始化步骤尚未全部完成并确认。"
                    "当前有活跃任务，可在任务完成后处理初始化，"
                    "或在本轮穿插完成初始化步骤（逐项确认 IDENTITY/SOUL/USER/TOOLS 内容是否落实），"
                    "完成后用 file.delete 删除 BOOTSTRAP.md 以结束引导阶段。"
                )
            else:
                content = (
                    "[初始化待完成] BOOTSTRAP.md 仍然存在，说明初始化检查项尚未全部完成并确认。"
                    "当前无活跃任务，这是推进初始化的自然时机："
                    "逐项确认 IDENTITY / SOUL / USER / TOOLS 的内容是否已具体落实，"
                    "完成后用 file.delete 删除 BOOTSTRAP.md 以结束引导阶段。"
                )
            loop._wm.add(WMItem(
                kind="bootstrap",
                content=content,
                priority=loop._cfg.thresholds.wm_pri_signal,
            ))

    return active_task


async def _inject_tick_side_signals(loop: Any, running_updates: list[dict[str, Any]]) -> None:
    loop._wm.clear(kinds={"run_monitor"})
    if running_updates:
        running_count = sum(1 for item in running_updates if item.get("status") == "running")
        finished_count = sum(1 for item in running_updates if item.get("status") in {"succeeded", "failed", "cancelled"})
        loop._wm.add(WMItem(
            kind="run_monitor",
            content=f"[Run 监控] running={running_count} finished={finished_count}",
            priority=loop._cfg.thresholds.wm_pri_monitor,
        ))
        for item in running_updates:
            crystal = str(item.get("crystal") or "").strip()
            if crystal:
                loop._wm.add(WMItem(
                    kind="progress_crystal",
                    content=f"[运行中结晶 run#{item.get('run_id')}] {crystal}",
                    priority=loop._cfg.thresholds.wm_pri_progress,
                ))
                loop._episodic.record_event("run_progress", {
                    "run_id": item.get("run_id"),
                    "task_id": item.get("task_id"),
                    "session_id": item.get("session_id"),
                    "excerpt": crystal,
                })

    for sig in await loop._task_store.due_signals():
        payload = sig.get("payload") or {}
        note = (payload.get("note") or "").strip()
        repeat_desc = f"每 {sig['repeat_secs']}s 重复" if sig.get("repeat_secs") else "一次性"
        parts = [
            (
                f"[调度触发 #{sig['id']}] {sig['title']}"
                f"({repeat_desc},已送达本轮上下文;是否响应由你决定。"
                "delivery 后该 signal 会由 runtime 自动推进/完成)"
            ),
        ]
        if note:
            parts.append(f"任务内容:{note}")
        loop._wm.add(WMItem(
            kind="scheduler",
            content="\n".join(parts),
            priority=loop._cfg.thresholds.wm_pri_signal,
        ))
        await loop._task_store.ack_signal(sig["id"])
        _log.info("[scheduler] signal fired: #%s %s", sig["id"], sig["title"])

    now = time.monotonic()
    if now - loop._last_heartbeat_at >= loop._cfg.loop.heartbeat_interval:
        heartbeat_path = loop._cfg.workspace_dir / "HEARTBEAT.md"
        if heartbeat_path.exists():
            try:
                heartbeat_md = heartbeat_path.read_text(encoding="utf-8").strip()
                if heartbeat_md:
                    loop._wm.add(WMItem(
                        kind="heartbeat",
                        content=f"[心跳自检]\n{heartbeat_md}",
                        priority=loop._cfg.thresholds.wm_pri_signal,
                    ))
                    _log.info("[heartbeat] 注入 WM,间隔 %ds", loop._cfg.loop.heartbeat_interval)
            except Exception:
                pass
        loop._last_heartbeat_at = now


async def _prepare_tick_judgment_state(
    loop: Any,
    active_task: Any,
    user_message: str,
) -> _TickJudgmentPrep:
    cfg = loop._cfg
    next_step_fulfilled: bool | None = None
    if loop._last_next_step:
        next_step_fulfilled = loop._last_act_progressful
    percept = await loop._perception.sense(
        loop._wm,
        active_task,
        user_message=user_message,
        last_next_step=loop._last_next_step,
        last_decision=loop._last_decision,
    )

    loop._episodic.record_event("perception", {
        "prediction_error": round(percept.prediction_error, 4),
        "workspace_dirty": percept.workspace_dirty,
        "multimodal_inputs": len(getattr(percept, "multimodal_inputs", [])),
        "wm_pressure": round(loop._wm.pressure, 4),
    })

    events_batch = loop._episodic.list_events_multi(["perception", "emotion"], limit=8)
    perception_events = events_batch["perception"]
    try:
        perception_replay = build_perception_replay(
            perception_events,
            high_error_threshold=cfg.thresholds.prediction_error_task,
            trend_delta=cfg.thresholds.perception_replay_trend_delta,
            high_error_hint_streak=cfg.thresholds.perception_replay_high_error_hint_streak,
        )
    except Exception:
        perception_replay = _perception_replay_fallback()

    if active_task is None:
        loop._idle_cycles += 1
    else:
        loop._idle_cycles = 0
        loop._last_curiosity_signal_idle_cycle = 0

    _model = loop._judgment.self_model
    _last_progress = loop._last_act_progressful
    _consecutive_no_progress = getattr(loop, '_consecutive_no_progress_count', 0)
    if not _last_progress and active_task is not None:
        _consecutive_no_progress += 1
    else:
        _consecutive_no_progress = 0
    loop._consecutive_no_progress_count = _consecutive_no_progress

    if _consecutive_no_progress >= 3 and active_task is not None:
        _cost_note = ""
        if _model.billing_mode == "token" and _model.estimated_cost_usd > 0:
            _cost_per_tick = _model.estimated_cost_usd / max(1, _model.tick_count)
            _cost_note = f"当前估算单次 Tick 成本 ${_cost_per_tick:.4f}。"
        _warning_msg = (
            f"[空转预警] 连续 {_consecutive_no_progress} 次操作未产生实质进展。"
            + _cost_note
            + "建议：1. 检查 next_step 是否过于模糊；2. 优先执行 file.read/exec 等低成本取证动作；3. 考虑 task.wait 或 pause。"
        )
        loop._wm.add(WMItem(
            kind="self_awareness",
            content=_warning_msg,
            priority=cfg.thresholds.wm_pri_critical,
        ))
        if loop._pending_idle_gap is None or loop._pending_idle_gap < 5.0:
            loop._pending_idle_gap = 5.0

    cognitive_signals = loop._perception.derive_cognitive_signals(
        percept,
        loop._wm,
        loop._emotion,
        cfg,
        has_active_task=active_task is not None,
        idle_cycles=loop._idle_cycles,
        next_step_fulfilled=next_step_fulfilled,
    )
    loop._behavior.apply_cognitive_probe(cognitive_signals)
    cognitive_signals.last_action_tool = loop._last_action_tool
    cognitive_signals.last_action_key = loop._last_action_key
    cognitive_signals.last_action_status = loop._last_action_status
    cognitive_signals.last_action_summary = loop._last_action_summary
    cognitive_signals.last_action_error = loop._last_action_error
    cognitive_signals.last_action_state_delta = loop._last_action_state_delta
    cognitive_signals.last_action_progressful = loop._last_act_progressful if loop._last_action_status else None
    cognitive_signals.last_action_progress_reason = loop._last_act_progress_reason if loop._last_action_status else ""
    cognitive_signals.recent_action_history = list(loop._recent_action_feedback)

    (failures_recent,) = await asyncio.gather(
        loop._task_store.list_failures(limit=5),
    )
    loop._emotion.derive_from_signals(
        failure_count=len(failures_recent),
        prediction_error=percept.prediction_error,
        wm_pressure=loop._wm.pressure,
        workspace_dirty=percept.workspace_dirty,
        alpha=cfg.emotion.ema_alpha,
        emotion_cfg=cfg.emotion,
        high_error_streak=perception_replay.high_error_streak,
        replay_trend=perception_replay.trend,
        has_active_task=active_task is not None,
        has_next_step=bool(active_task and active_task.next_step),
        task_status=active_task.status if active_task else "",
    )

    loop._episodic.record_event("emotion", {
        "valence": round(loop._emotion.valence, 4),
        "arousal": round(loop._emotion.arousal, 4),
        "dominance": round(loop._emotion.dominance, 4),
        "dominant": loop._emotion.dominant,
        "regulation_strategy": loop._emotion.regulation.strategy,
        "regulation_reason": loop._emotion.regulation.reason,
    })

    emotion_replay = build_emotion_replay(
        events_batch["emotion"],
        trend_delta=cfg.thresholds.emotion_replay_trend_delta,
    )

    ethos_baseline_json, _ = await loop._task_store.get_fact("soul:ethos_baseline")
    ethos_baseline: EthosValues | None = None
    if ethos_baseline_json:
        try:
            ethos_baseline = EthosValues.from_dict(json.loads(ethos_baseline_json))
        except (ValueError, json.JSONDecodeError) as _ethos_exc:
            _log.warning("[tick] ethos_baseline 解析失败，使用 config 默认值: %s", _ethos_exc)
    ethos_state = derive_ethos_state(
        failure_count=len(failures_recent),
        high_error_streak=perception_replay.high_error_streak,
        has_active_task=active_task is not None,
        has_next_step=bool(active_task and active_task.next_step),
        perception_trend=perception_replay.trend,
        emotion_down_regulate_streak=emotion_replay.down_regulate_streak,
        ethos_cfg=cfg.soul.ethos,
        baseline=ethos_baseline,
    )

    _log.debug(
        "[tick] emotion=%s v=%.2f a=%.2f | ethos truth=%.2f caution=%.2f curiosity=%.2f",
        loop._emotion.dominant,
        loop._emotion.valence,
        loop._emotion.arousal,
        ethos_state.values.truth,
        ethos_state.values.caution,
        ethos_state.values.curiosity,
    )

    signals = compute_judgment_signals(
        failure_count=len(failures_recent),
        high_error_streak=perception_replay.high_error_streak,
        perception_trend=perception_replay.trend,
        emotion_state=loop._emotion,
        thresholds=cfg.thresholds,
    )
    constitution_text = load_constitution(cfg.constitution_path)
    hard_boundaries = extract_constitution_boundaries(constitution_text)
    return _TickJudgmentPrep(
        percept=percept,
        perception_replay=perception_replay,
        cognitive_signals=cognitive_signals,
        ethos_state=ethos_state,
        signals=signals,
        hard_boundaries=hard_boundaries,
    )


async def _decide_initial_action(
    loop: Any,
    cycle: int,
    user_message: str,
    active_task: Any,
    chat_id: str | None,
    prep: _TickJudgmentPrep,
) -> Any:
    cfg = loop._cfg
    has_llm_wake_signal = any(
        item.get("kind") in _LLM_WAKE_WM_KINDS for item in loop._wm.get_top(20)
    )
    skip_llm = (
        cfg.loop.judge_every > 1
        and not user_message
        and active_task is None
        and not has_llm_wake_signal
        and loop._ticks_since_judge < cfg.loop.judge_every - 1
    )
    if skip_llm:
        loop._ticks_since_judge += 1
        _log.debug(
            "[loop] tick=%d 跳过 LLM 判断(聚合 %d/%d)",
            cycle,
            loop._ticks_since_judge,
            cfg.loop.judge_every,
        )
        from core.judgment import JudgmentOutput

        return JudgmentOutput.wait(
            reason=f"[按请求聚合] 空闲跳过 LLM({loop._ticks_since_judge}/{cfg.loop.judge_every})"
        )

    pending_initial_thinking = loop._pending_thinking_override
    if user_message:
        chat_floor = cfg.loop.chat_thinking if cfg.loop.chat_thinking != cfg.thinking else None
        pending_initial_thinking = _thinking_floor(pending_initial_thinking, chat_floor)
    thinking_override = _resolve_thinking_override(
        cfg,
        user_message=user_message,
        pending_override=pending_initial_thinking,
    )
    from core.loop.runtime.life import collect_runtime_life_snapshot

    runtime_life_snapshot = collect_runtime_life_snapshot(loop).as_dict()
    action = await loop._judgment.decide(
        prep.percept,
        loop._wm,
        loop._task_store,
        loop._episodic,
        loop._semantic,
        loop._emotion,
        active_task=active_task,
        user_message=user_message,
        chat_id=chat_id,
        ethos_state=prep.ethos_state,
        judgment_signals=prep.signals,
        hard_boundaries=prep.hard_boundaries,
        perception_replay=prep.perception_replay,
        cognitive_signals=prep.cognitive_signals,
        thinking_override=thinking_override,
        phase="initial",
        prefer_tier=_prefer_tier_for_task(
            loop._pending_tier,
            active_task,
            has_user_message=bool(user_message),
        ),
        routing_overrides=loop._pending_routing_overrides,
        runtime_life_snapshot=runtime_life_snapshot,
    )
    loop._pending_tier = None
    loop._pending_thinking_override = None
    loop._ticks_since_judge = 0
    return action


async def _review_delegate_tasks(
    loop: Any,
    ctx: Any,
    action: Any,
    user_message: str,
    active_task: Any,
) -> Any:
    if not action.delegate_tasks:
        return action

    from core.loop.task.parallel import run_tasks_parallel

    parent_task_id = active_task.id if active_task else None
    parallel_entries = await run_tasks_parallel(action.delegate_tasks, ctx, loop, parent_task_id)

    for entry in parallel_entries:
        loop._wm.add(WMItem(
            kind="task_result",
            content=entry.get("summary", ""),
            priority=loop._cfg.thresholds.wm_pri_user_msg,
        ))

    _log.info(
        "[loop] delegate gate review: %d task results ids=%s",
        len(parallel_entries),
        [entry.get("tool", "") for entry in parallel_entries],
    )
    result_action = await loop._judgment.decide_continue(
        tool_history=parallel_entries or [{
            "tool": "delegate",
            "params": {},
            "result": "无有效子任务",
            "status": "ok",
            "error": "",
        }],
        user_message=user_message,
        active_task=active_task,
        prefer_tier="reasoner",
    )
    # 防止 decide_continue 仍返回 delegate_tasks（无限委托）
    if result_action.delegate_tasks and not result_action.chosen_action_id:
        _log.warning(
            "[loop] decide_continue after delegate returned another delegate, converting to wait"
        )
        from core.judgment import JudgmentOutput  # noqa: PLC0415
        return JudgmentOutput.wait("delegate 续判仍为 delegate，转为等待")
    return result_action


def _log_tick_decision(loop: Any, cycle: int, action: Any) -> None:
    cfg = loop._cfg
    loop._judgment.self_model.record_tick()
    loop._judgment.self_model.record_api_call()
    call_meta = loop._judgment.last_call_meta
    actual_model = call_meta.get("model_ref") or cfg.model
    actual_thinking = call_meta.get("thinking") or cfg.thinking
    actual_tier = call_meta.get("tier") or "default"
    actual_phase = call_meta.get("phase") or "initial"
    actual_skills = call_meta.get("skills") or "none"
    usage_source = call_meta.get("usage_source")
    action_label = action.action_label() or action.decision or "-"
    scope = tick_scope_fields(
        tick=cycle,
        decision=action.decision,
        tool=action_label,
        model_ref=actual_model,
        tier=actual_tier,
        phase=actual_phase,
        thinking=actual_thinking,
        skills=actual_skills,
        usage_source=usage_source,
    )
    console.print(
        f"[bold cyan][loop][/bold cyan] {scope} rationale={action.rationale or ''}"
    )
    _log.info(
        "[loop] %s rationale=%s",
        scope,
        action.rationale or "",
    )


class _TickPerceptionPhase:
    @staticmethod
    async def run(
        loop: Any,
        user_message: str,
        chat_id: str | None,
    ) -> tuple[_TickJudgmentPrep, Any]:
        running_updates = await refresh_running_runs(
            loop._task_store,
            episodic=loop._episodic,
            semantic=loop._semantic,
            metabolic=_loop_metabolic(loop),
        )
        active_task = await _prepare_active_task_for_tick(loop, user_message, chat_id)
        await _inject_tick_side_signals(loop, running_updates)
        if user_message:
            _dropped = loop._wm.salience_gate(
                user_message,
                preserve_kinds={"bootstrap_identity", "self_awareness", "task_anchor"},
                priority_floor=loop._cfg.thresholds.wm_pri_signal,
            )
            if _dropped:
                _log.debug("[wm-gate] salience_gate dropped %d low-relevance items", _dropped)
        prep = await _prepare_tick_judgment_state(loop, active_task, user_message)
        return prep, active_task


class _TickJudgmentPhase:
    @staticmethod
    async def run(
        loop: Any,
        ctx: Any,
        cycle: int,
        user_message: str,
        active_task: Any,
        chat_id: str | None,
        prep: _TickJudgmentPrep,
    ) -> Any:
        if active_task is None:
            await loop._maybe_curiosity_task(prep.ethos_state)
        if active_task is not None:
            plan = (getattr(active_task, "extras", None) or {}).get("plan")
            if isinstance(plan, list):
                in_progress_step = next(
                    (
                        str(item.get("step") or "").strip()
                        for item in plan
                        if isinstance(item, dict) and str(item.get("status") or "").strip() == "in_progress"
                    ),
                    None,
                )
                current_step = str(getattr(active_task, "current_step", "") or "").strip()
                if in_progress_step and current_step != in_progress_step:
                    loop._wm.add(WMItem(
                        kind="self_awareness",
                        content=(
                            f"[计划对齐] task.plan 进行中步骤「{in_progress_step}」，"
                            f"task.current_step 为「{current_step or '（未设置）'}」。"
                        ),
                        priority=loop._cfg.thresholds.wm_pri_wait_aware,
                    ))
        action = await _decide_initial_action(loop, cycle, user_message, active_task, chat_id, prep)
        _log_tick_decision(loop, cycle, action)
        action = await _review_delegate_tasks(loop, ctx, action, user_message, active_task)
        if action.decision == "act" and action.reply_to_user:
            action.speech_intent = action.reply_to_user
            action.reply_to_user = ""
        return action
