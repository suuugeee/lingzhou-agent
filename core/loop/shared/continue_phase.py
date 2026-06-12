"""core.loop.shared.continue_phase — tick 内续判工具循环。

从 tick.py 分离，避免 tick.py 超长。
由 tick() 在初始动作执行后、满足续判条件时调用。
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from core.judgment.context.utils import _clip_for_context
from memory.working import WMItem

from .common import (
    _maybe_reconcile_bootstrap,
    _next_initial_tier_hint,
    _resolve_thinking_override,
    _should_continue_within_tick,
    _tool_history_entry,
)
from .progress import action_key_param

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput

_log = logging.getLogger("lingzhou.loop")

_REPEAT_COMPACT_TOOLS = frozenset({
    "file.read",
    "file.list",
    "memory.search",
    "task.list",
    "probe.run",
    "web.fetch",
    "shell.run",
})


def _registry_has_tool(registry: Any | None, tool_name: str) -> bool:
    getter = getattr(registry, "get", None)
    if getter is None:
        return False
    try:
        return getter(tool_name) is not None
    except Exception:
        return False


def _compact_history_line(entry: dict[str, Any]) -> str:
    tool = str(entry.get("tool") or "?")
    status = str(entry.get("status") or "?")
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    key = (
        params.get("path")
        or params.get("name")
        or params.get("title")
        or params.get("key")
        or str(params.get("id") or "")
        or params.get("command")
        or params.get("query")
        or entry.get("resource_key")
        or ""
    )
    result = str(entry.get("result") or entry.get("summary") or "").strip()
    result_preview = _clip_for_context(result, 150) if result else ""  # 优化：缩短摘要长度以降低上下文压力
    result_hash = hashlib.md5(result.encode("utf-8", errors="replace")).hexdigest()[:12] if result else ""
    meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    log_summary = str(meta.get("log_summary") or "").strip()
    summary_preview = _clip_for_context(log_summary, 150) if log_summary else result_preview  # 优化：缩短摘要长度以降低上下文压力
    artifacts = [str(item) for item in (entry.get("artifact_paths") or []) if item]
    raw_state_delta = entry.get("state_delta")
    if isinstance(raw_state_delta, dict):
        clipped_state_delta = {
            str(_clip_for_context(str(key), 64)): _clip_for_context(str(value), 220)
            for key, value in raw_state_delta.items()
        }
    else:
        clipped_state_delta = {}
    facts: dict[str, Any] = {
        "tool": tool,
        "status": status,
    }
    if key:
        facts["key"] = str(key)
    if entry.get("error"):
        facts["error"] = str(entry.get("error"))
    if log_summary:
        facts["summary"] = summary_preview
    elif result:
        facts["summary_preview"] = result_preview
        facts["result_chars"] = len(result)
        facts["result_hash"] = result_hash
    if entry.get("fingerprint"):
        facts["fingerprint"] = str(entry.get("fingerprint"))
    if artifacts:
        facts["artifacts"] = artifacts
    if clipped_state_delta:
        facts["state_delta"] = clipped_state_delta
    return json.dumps(facts, ensure_ascii=False, sort_keys=True)


def _repeat_history_signature(entry: dict[str, Any]) -> str:
    tool = str(entry.get("tool") or "")
    if tool not in _REPEAT_COMPACT_TOOLS:
        return ""
    if entry.get("error") or entry.get("skipped") or str(entry.get("status") or "") not in {"", "ok"}:
        return ""
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    params_text = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    result = str(entry.get("result") or entry.get("summary") or "")
    if not result:
        return ""
    result_hash = hashlib.md5(result.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{tool}|{params_text}|{result_hash}"


def _action_history_key(action: JudgmentOutput) -> tuple[str, str]:
    return str(action.chosen_action_id or ""), action_key_param(action.params)


def _trailing_same_action_count(history: list[dict[str, Any]], tool_name: str, key_param: str) -> int:
    if not tool_name:
        return 0
    count = 0
    for entry in reversed(history):
        tool = str(entry.get("tool") or "")
        if tool == "[repeat-compacted]":
            params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
            if params.get("repeat_tool") == tool_name and params.get("repeat_key") == key_param:
                count += int(params.get("repeat_count") or 0)
                continue
            break
        if tool.startswith("["):
            continue
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        entry_key = action_key_param(params)
        if tool != tool_name or entry_key != key_param:
            break
        count += 1
    return count


def _continue_repeat_threshold(loop: Any) -> int:
    cfg = getattr(loop, "_cfg", None)
    loop_cfg = getattr(cfg, "loop", None)
    return max(2, int(getattr(loop_cfg, "behavior_streak_threshold", 3) or 3))


def _build_continue_repeat_workbench_action(
    *,
    action: JudgmentOutput,
    tool_name: str,
    key_param: str,
    repeat_count: int,
) -> JudgmentOutput:
    from core.judgment import JudgmentOutput  # noqa: PLC0415

    return JudgmentOutput(
        decision="act",
        chosen_action_id="task.workbench",
        params={
            "workbench": {
                "domain": "runtime-loop",
                "intent": "continue 阶段停止重复同一取证动作",
                "evidence": [
                    f"本 tick 内 {tool_name} {key_param or '（空参数）'} 已连续出现 {repeat_count} 次。",
                    "继续执行同一工具和同一关键参数不会增加有效证据，只会扩大上下文和工具历史。",
                ],
                "hypothesis": "当前卡点不是缺少再次读取，而是需要综合已有结果或切换到更高信息增量的验证方式。",
                "recovery_state": "continue_repeat_action_gated",
                "next_verification": (
                    f"不要再重复执行 {tool_name} {key_param or ''}；"
                    "先总结已有证据，或换用不同工具/参数验证同一假设。"
                ),
                "completion_checks": [
                    "已停止本 tick 内重复工具调用。",
                    "已把重复动作转化为明确的下一步验证约束。",
                ],
            }
        },
        rationale=(
            f"continue 行为门控改道：{tool_name} {key_param or '（空参数）'} "
            f"在同一 tick 内已连续重复 {repeat_count} 次，先写工作台收敛。"
        ),
        reflection=action.reflection,
        next_step="先综合已有证据；仍需验证时换不同证据源。",
        model_strategy=dict(action.model_strategy or {}),
        applied_skills=list(action.applied_skills or []),
    )


def _compact_repeated_tool_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """原地合并本 tick 内重复的低价值成功工具结果，保留最新一次完整记录。"""
    if len(history) < 2:
        return history
    counts: dict[str, int] = {}
    latest_index: dict[str, int] = {}
    for index, entry in enumerate(history):
        sig = _repeat_history_signature(entry)
        if not sig:
            continue
        counts[sig] = counts.get(sig, 0) + 1
        latest_index[sig] = index
    repeated = {sig for sig, count in counts.items() if count > 1}
    if not repeated:
        return history

    old_lines: dict[str, list[str]] = {sig: [] for sig in repeated}
    rebuilt: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for index, entry in enumerate(history):
        sig = _repeat_history_signature(entry)
        if sig not in repeated:
            rebuilt.append(entry)
            continue
        if index != latest_index[sig]:
            old_lines[sig].append(_compact_history_line(entry))
            continue
        if sig not in emitted:
            omitted = counts[sig] - 1
            params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
            rebuilt.append({
                "tool": "[repeat-compacted]",
                "params": {
                    "repeat_tool": str(entry.get("tool") or ""),
                    "repeat_key": action_key_param(params),
                    "repeat_count": omitted,
                },
                "result": (
                    f"（本 tick 内 {omitted} 条重复低价值工具调用已压缩；"
                    "最新一次完整结果保留在下一条，原始记录保留在 run/artifact 中）\n"
                    + "\n".join(old_lines.get(sig) or [])
                ),
                "status": "compacted",
                "error": "",
            })
            emitted.add(sig)
        rebuilt.append(entry)
    history[:] = rebuilt
    return history


def _compact_tool_history(history: list[dict[str, Any]], *, keep_last: int) -> list[dict[str, Any]]:
    """原地压缩早期 tool_history，避免上下文爆炸且不丢失外层列表引用。"""
    keep_last = max(1, int(keep_last))
    if len(history) <= keep_last:
        return history
    older = history[:-keep_last]
    recent = history[-keep_last:]
    summary_lines = []
    for entry in older:
        summary_lines.append(_compact_history_line(entry))
    compact: dict[str, Any] = {
        "tool": "[compacted]",
        "params": {},
        "result": f"（早期 {len(older)} 条工具调用已结构化压缩；原始结果保留在 run/artifact 中）\n" + "\n".join(summary_lines),
        "status": "compacted",
        "error": "",
    }
    history[:] = [compact] + recent
    return history


async def _record_continue_round_limit(
    *,
    loop: Any,
    ctx: Any,
    active_task: Any,
    tool_history: list[dict[str, Any]],
    max_inner_rounds: int,
) -> tuple[JudgmentOutput | None, Any | None]:
    if active_task is None or not _registry_has_tool(loop._registry, "task.workbench"):
        return None, None
    from core.judgment import JudgmentOutput

    recent_tools = [
        str(entry.get("tool") or "")
        for entry in tool_history[-max(1, min(6, len(tool_history))):]
        if str(entry.get("tool") or "")
    ]
    action = JudgmentOutput(
        decision="act",
        chosen_action_id="task.workbench",
        params={
            "workbench": {
                "domain": "runtime-loop",
                "intent": "continue 阶段达到单 tick 工具续判上限，收敛到下一轮验证",
                "evidence": [
                    f"本 tick continue 阶段已执行 {max_inner_rounds} 轮工具续判。",
                    f"最近工具序列: {', '.join(recent_tools) if recent_tools else '（无）'}",
                ],
                "hypothesis": "当前任务仍需推进，但继续留在同一 tick 内追加工具会削弱总结与用户可见收敛。",
                "recovery_state": "continue_round_limit_reached",
                "next_verification": "下一轮先综合本 tick 工具结果，确认是否已经足够回答/完成；若不足，再选择一个最高信息增量的验证动作。",
                "completion_checks": [
                    "已停止在同一 tick 内继续追加工具调用。",
                    "已把本轮工具结果收敛为下一轮的验证入口。",
                ],
            }
        },
        rationale=(
            f"continue 阶段达到 {max_inner_rounds} 轮上限，先写入任务皮层收敛状态，"
            "避免单 tick 内无限续判。"
        ),
        next_step="下一轮先综合本 tick 工具结果，再决定是否继续取证。",
    )
    result = await loop._execution.dispatch(action, ctx)
    tool_history.append(_tool_history_entry(action, result))
    return action, result


async def _run_continue_phase(
    *,
    loop: Any,
    ctx: Any,
    user_message: str,
    active_task: Any,
    cognitive_signals: Any,
    action: JudgmentOutput,
    result: Any,
    tool_history: list[dict[str, Any]],
) -> tuple[JudgmentOutput, Any]:
    """执行 continue 阶段的工具续判循环。

    返回 (final_action, final_result)，调用方用这两个值替换初始阶段的 action/result。
    tool_history 通过引用追加（in-place mutation）。
    """
    cfg = loop._cfg
    affect = {"valence": loop._emotion.valence, "arousal": loop._emotion.arousal}
    _wm_delta: list[dict] = []  # 本 tick continue 阶段新增的 WM 条目（不在 tool_history 里）
    from core.judgment.policy import tool_history_compact_limits

    compact_threshold, keep_last = tool_history_compact_limits(cfg)
    max_inner_rounds = max(1, int(getattr(cfg.thresholds, "continue_max_inner_rounds", 4) or 4))

    _inner = 0
    while True:
        if _inner >= max_inner_rounds:
            recorded_action, recorded_result = await _record_continue_round_limit(
                loop=loop,
                ctx=ctx,
                active_task=active_task,
                tool_history=tool_history,
                max_inner_rounds=max_inner_rounds,
            )
            if recorded_action is not None and recorded_result is not None:
                action = recorded_action
                result = recorded_result
            else:
                _log.warning(
                    "[continue] reached max inner rounds=%d, breaking without workbench",
                    max_inner_rounds,
                )
            break

        if await loop._task_store.has_pending_chat_message():
            _log.debug("[continue] chat 消息到达，中断工具循环 inner=%d", _inner)
            break

        # 工具历史超长时压缩早期条目，避免上下文窗口爆炸
        _compact_repeated_tool_history(tool_history)
        if len(tool_history) >= compact_threshold and len(tool_history) > keep_last:
            _compact_tool_history(tool_history, keep_last=keep_last)

        next_tier = _next_initial_tier_hint(action) or ""
        continue_thinking = _resolve_thinking_override(
            cfg,
            user_message=user_message,
            model_strategy=action.model_strategy,
        )
        cont = await loop._judgment.decide_continue(
            tool_history,
            user_message=user_message,
            active_task=active_task,
            prefer_tier=next_tier or None,
            thinking_override=continue_thinking,
            routing_overrides=loop._pending_routing_overrides,
            wm_delta=_wm_delta or None,
        )

        if (
            cont.decision == "act"
            and cont.chosen_action_id
            and cont.chosen_action_id != "task.workbench"
            and _registry_has_tool(loop._registry, "task.workbench")
        ):
            tool_name, key_param = _action_history_key(cont)
            repeat_count = _trailing_same_action_count(tool_history, tool_name, key_param) + 1
            if repeat_count >= _continue_repeat_threshold(loop):
                cont = _build_continue_repeat_workbench_action(
                    action=cont,
                    tool_name=tool_name,
                    key_param=key_param,
                    repeat_count=repeat_count,
                )

        if cont.decision == "act":
            tool_name = cont.chosen_action_id or ""
            key_param = action_key_param(cont.params)
            behavior = getattr(loop, "_behavior", None)
            on_act = getattr(behavior, "on_act", None)
            if callable(on_act):
                for behavior_item in on_act(
                    tool_name,
                    key_param,
                    str(active_task.id) if active_task else None,
                    cont.params,
                ):
                    loop._wm.add(behavior_item)
                    _wm_delta.append(behavior_item.to_dict())
            apply_probe = getattr(behavior, "apply_cognitive_probe", None)
            if callable(apply_probe) and cognitive_signals is not None:
                apply_probe(cognitive_signals)
        if cognitive_signals is not None:
            cognitive_signals.active_task_id = getattr(active_task, "id", "") if active_task is not None else ""
            cognitive_signals.active_task_source = getattr(active_task, "source", "") if active_task is not None else ""
            cognitive_signals.active_task_status = getattr(active_task, "status", "") if active_task is not None else ""
            cognitive_signals.active_task_next_step = getattr(active_task, "next_step", "") if active_task is not None else ""
        behavior = getattr(loop, "_behavior", None)
        gate = getattr(behavior, "apply_execution_gate", None)
        if gate is not None and cognitive_signals is not None:
            gated = gate(cont, cognitive_signals)
            if gated is not cont:
                cont = gated
                tool_name = ""
                key_param = ""
        cont_result = None

        # 防止 decide_continue 返回 delegate_tasks 但无 chosen_action_id 时派发空工具
        if cont.decision == "act" and not cont.chosen_action_id and cont.delegate_tasks:
            _log.warning(
                "[continue] decide_continue returned delegate_tasks without chosen_action_id, skipping dispatch"
            )
            from core.judgment import JudgmentOutput  # noqa: PLC0415
            cont = JudgmentOutput.wait("continue 阶段 delegate_tasks 无工具名，转为等待")

        if cont_result is None:
            cont_result = await loop._execution.dispatch(cont, ctx)

        if cont_result.summary and (not cont_result.skipped or cont_result.error):
            tool_name = cont.chosen_action_id or ""
            key_param = action_key_param(cont.params)
            prefix = f"[{tool_name}{'  ' + key_param if key_param else ''}] "
            _result_item = WMItem(
                kind=tool_name or cont_result.kind,
                content=prefix + cont_result.summary,
                priority=cont_result.priority,
            )
            loop._wm.add(_result_item)
            # 不论成功还是 error-skipped，均追加到 wm_delta，让后续 continue 轮可感知
            _wm_delta.append(_result_item.to_dict())
        if cont.reflection and cont.reflection.strip():
            loop._wm.add(WMItem(
                kind="synthesis",
                content=f"[合成] {cont.reflection.strip()}",
                priority=loop._cfg.thresholds.wm_pri_insight,
            ))
        if cont.rationale:
            loop._episodic.record(
                role="assistant",
                content=f"[inner-{_inner + 1}] {cont.rationale}",
                task_id=str(active_task.id) if active_task else None,
                affect=affect,
            )

        if cont.decision == "act":
            behavior = getattr(loop, "_behavior", None)
            if cont_result.error and "oldtextnotfound" in (cont_result.error or "").lower():
                on_edit_failure = getattr(behavior, "on_edit_failure", None)
                if callable(on_edit_failure):
                    for behavior_item in on_edit_failure(cont_result.error or ""):
                        loop._wm.add(behavior_item)
            on_act_result = getattr(behavior, "on_act_result", None)
            if callable(on_act_result):
                on_act_result(cont.chosen_action_id or "", cont_result.summary or "")
            tool_history.append(_tool_history_entry(cont, cont_result))

        if cont.reply_to_user:
            cont.speech_intent = cont.reply_to_user
            cont.reply_to_user = ""

        action = cont
        result = cont_result
        _inner += 1
        if action.speech_intent or not _should_continue_within_tick(
            action,
            user_message=user_message,
            has_active_task=active_task is not None,
            registry=loop._registry,
            result=result,
        ):
            break
        # PlanUnchanged：计划结构没变，继续循环只会死锁；跳出让下一 tick 直接执行具体工具。
        if cont_result.error == "PlanUnchanged":
            _log.debug("[continue] PlanUnchanged — 跳出 continue 循环，下一 tick 直接执行")
            break

    # ② continue 循环结束后同步检测（兜底 inner 轮删除 BOOTSTRAP.md 的场景）
    await _maybe_reconcile_bootstrap(loop)

    return action, result
