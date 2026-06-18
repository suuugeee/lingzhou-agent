"""core.loop.shared.continue_phase — tick 内续判工具循环。

从 tick.py 分离，避免 tick.py 超长。
由 tick() 在初始动作执行后、满足续判条件时调用。
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from core.cortex.actions import build_workbench_action
from core.cortex import intent as cortex_intent
from core.judgment.context.utils import _clip_for_context
from memory.working import WMItem
from tools.registry import registry_has_tool

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
_LOW_INCREMENT_CONTINUE_TOOLS = frozenset({
    "file.read",
    "file.list",
    "memory.search",
    "task.list",
    "probe.list",
    "probe.run",
    "config.list_keys",
})
_TASK_SCOPED_CONTINUE_TOOLS = frozenset({
    "task.advance",
    "task.update",
    "task.workbench",
    "task.complete",
    "task.fail",
    "task.wait",
})


def _switch_hints(tool_name: str, key_param: str) -> str:
    target = f"{key_param}" if key_param else "当前对象"
    if tool_name == "file.read":
        if key_param:
            return (
                f"不要再重复读取 {target}；基于已读结果先形成结论，"
                "或改用 shell.run/grep 对该路径做定位验证。"
            )
        return "不要继续重复同类文件读取；改用 shell.run 或 task.workbench 切换验证层。"
    if tool_name == "file.list":
        if key_param:
            return (
                f"不要再重复列目录 {target}；从现有目录结果选择具体文件，"
                "改做 file.read 或 grep。"
            )
        return "不要继续重复目录枚举；改为定位具体文件后读取或直接执行 task.workbench 验证。"
    if tool_name == "memory.search":
        return "不要重复同一 query 的 memory.search；改为读取命中语义 ID 或切换到 shell.run/file.read。"
    if tool_name == "probe.run":
        return "不要重复执行同一探针；改为切换到一个可验证的任务动作。"
    if tool_name == "shell.run":
        return "避免重复 shell.run 同构命令；改为 task.workbench 复核 next_verification，或执行一条更聚焦验证命令。"
    return f"避免继续重复 {target}；切换到一个更高信息增量动作。"


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


def _compacted_history_entry(tool: str, result: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"tool": tool, "params": params or {}, "result": result, "status": "compacted", "error": ""}


def _action_history_key(action: JudgmentOutput) -> tuple[str, str]:
    return str(action.chosen_action_id or ""), action_key_param(action.params)


def _coerce_task_id(value: Any) -> int | None:
    try:
        task_id = int(value)
    except (TypeError, ValueError):
        return None
    return task_id if task_id > 0 else None


def _latest_task_id_from_history(history: list[dict[str, Any]]) -> int | None:
    for entry in reversed(history):
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        state_delta = entry.get("state_delta") if isinstance(entry.get("state_delta"), dict) else {}
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        for raw in (
            params.get("task_id"),
            state_delta.get("task_id"),
            metadata.get("task_id"),
            entry.get("resource_key"),
        ):
            task_id = _coerce_task_id(raw)
            if task_id is not None:
                return task_id
    return None


def _ensure_continue_task_id(action: JudgmentOutput, active_task: Any | None, history: list[dict[str, Any]]) -> None:
    if action.decision != "act" or action.chosen_action_id not in _TASK_SCOPED_CONTINUE_TOOLS:
        return
    if not isinstance(action.params, dict):
        action.params = {}
    if _coerce_task_id(action.params.get("task_id")) is not None:
        return
    task_id = _coerce_task_id(getattr(active_task, "id", None))
    if task_id is None:
        task_id = _latest_task_id_from_history(history)
    if task_id is not None:
        action.params["task_id"] = task_id


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
    return max(2, int(getattr(loop_cfg, "behavior_streak_threshold", 4) or 4))


def _continue_low_increment_budget(loop: Any, max_inner_rounds: int) -> int:
    cfg = getattr(loop, "_cfg", None)
    thresholds = getattr(cfg, "thresholds", None)
    explicit = getattr(thresholds, "continue_low_increment_budget", None) if thresholds is not None else None
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    return max(3, min(4, int(max_inner_rounds)))


def _low_increment_history_count(history: list[dict[str, Any]]) -> int:
    count = 0
    for entry in history:
        tool = str(entry.get("tool") or "")
        if tool in _LOW_INCREMENT_CONTINUE_TOOLS:
            count += 1
            continue
        if tool == "[repeat-compacted]":
            params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
            repeat_tool = str(params.get("repeat_tool") or "")
            if repeat_tool in _LOW_INCREMENT_CONTINUE_TOOLS:
                count += int(params.get("repeat_count") or 0)
    return count


def _recent_tool_names(history: list[dict[str, Any]], *, limit: int = 6) -> list[str]:
    return [
        str(entry.get("tool") or "")
        for entry in history[-max(1, min(limit, len(history))):]
        if str(entry.get("tool") or "")
    ]


def _build_continue_repeat_workbench_action(
    *,
    action: JudgmentOutput,
    tool_name: str,
    key_param: str,
    repeat_count: int,
) -> JudgmentOutput:
    workbench = {
        "domain": "runtime-loop",
        "intent": "continue 阶段停止重复同一取证动作",
        "evidence": [
            f"本 tick 内 {tool_name} {key_param or '（空参数）'} 已连续出现 {repeat_count} 次。",
            "继续执行同一工具和同一关键参数不会增加有效证据，只会扩大上下文和工具历史。",
        ],
        "hypothesis": "当前卡点不是缺少再次读取，而是需要综合已有结果或切换到更高信息增量的验证方式。",
        "recovery_state": "continue_repeat_action_gated",
        "next_verification": cortex_intent.control_next_verification(
            f"{_switch_hints(tool_name, key_param)} "
            "若证据仍不足，先形成可验证结论再提交下一个单点动作。"
        ),
        "completion_checks": [
            "已停止本 tick 内重复工具调用。",
            "已把重复动作转化为明确的下一步验证约束。",
        ],
    }
    return build_workbench_action(
        workbench=workbench,
        rationale=(
            f"continue 行为门控改道：{tool_name} {key_param or '（空参数）'} "
            f"在同一 tick 内已连续重复 {repeat_count} 次，先写工作台收敛。"
        ),
        source_action=action,
        next_step="先综合已有证据；仍需验证时换不同证据源。",
    )


def _build_continue_low_increment_budget_action(
    *,
    action: JudgmentOutput,
    tool_name: str,
    budget: int,
    history: list[dict[str, Any]],
) -> JudgmentOutput:
    recent_tools = _recent_tool_names(history)
    workbench = {
        "domain": "runtime-loop",
        "intent": "continue 阶段停止低信息探索串联",
        "evidence": [
            f"本 tick continue 阶段低信息探索动作已达到预算 {budget} 次。",
            f"本轮候选动作仍是 {tool_name}。",
            f"最近工具序列: {', '.join(recent_tools) if recent_tools else '（无）'}",
        ],
        "hypothesis": "继续追加 list/read/search/probe 会扩大上下文压力；当前应先综合已有证据或切换到更高信息增量验证。",
        "recovery_state": "continue_low_increment_budget_reached",
        "next_verification": cortex_intent.control_next_verification(
            f"{_switch_hints(tool_name, action_key_param(action.params))} "
            "先形成收敛结论后，优先提交 task.workbench 或明确下一步单点任务动作。"
        ),
        "completion_checks": [
            "已停止同 tick 内连续低信息探索。",
            "已把已有结果收敛为结论或更具体的下一步验证。",
        ],
    }
    return build_workbench_action(
        workbench=workbench,
        rationale=(
            f"continue 行为门控改道：低信息探索动作已达到预算 {budget} 次，"
            f"本轮不再执行 {tool_name}，先写工作台收敛。"
        ),
        source_action=action,
        next_step="先综合已有证据；仍需验证时选择更高信息增量动作。",
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
            rebuilt.append(_compacted_history_entry(
                "[repeat-compacted]",
                (
                    f"（本 tick 内 {omitted} 条重复低价值工具调用已压缩；"
                    "最新一次完整结果保留在下一条，原始记录保留在 run/artifact 中）\n"
                    + "\n".join(old_lines.get(sig) or [])
                ),
                {
                    "repeat_tool": str(entry.get("tool") or ""),
                    "repeat_key": action_key_param(params),
                    "repeat_count": omitted,
                },
            ))
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
    summary_lines = [_compact_history_line(entry) for entry in older]
    compact = _compacted_history_entry(
        "[compacted]",
        f"（早期 {len(older)} 条工具调用已结构化压缩；原始结果保留在 run/artifact 中）\n" + "\n".join(summary_lines),
    )
    history[:] = [compact] + recent
    return history


def _specific_round_limit_next_verification(tool_history: list[dict[str, Any]]) -> str:
    """Choose a concrete recovery action when continue rounds hit the cap."""
    fallback = "根据最近一次工具结果选择一个具体验证动作；若已有足够证据，直接面向用户收敛答复或完成任务。"
    for entry in reversed(tool_history):
        if not isinstance(entry, dict):
            continue
        state_delta = entry.get("state_delta") if isinstance(entry.get("state_delta"), dict) else {}
        for key in ("recovery_next_step", "next_verification"):
            value = _clip_for_context(str(state_delta.get(key) or ""), 240)
            if value and not cortex_intent.is_control_next_verification(value):
                return value
        tool = str(entry.get("tool") or "").strip()
        error = str(entry.get("error") or "").strip()
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        key_param = action_key_param(params)
        if error == "ToolInputInvalid":
            missing = state_delta.get("missing_params") if isinstance(state_delta, dict) else None
            missing_text = ", ".join(str(item) for item in missing) if isinstance(missing, list) else ""
            if tool and missing_text:
                return f"按 {tool} 的 manifest 重新调用工具；补齐必填参数 {missing_text}。"
            if tool:
                return f"按 {tool} 的 manifest 修正参数后重试一次。"
        if error:
            return f"修复最近一次 {tool or '工具'} 失败（{_clip_for_context(error, 80)}），再用不同证据路径验证任务是否推进。"
        if str(entry.get("status") or "") == "ok" and tool and not tool.startswith("task."):
            summary = _clip_for_context(str(entry.get("summary") or entry.get("result") or ""), 140)
            if summary:
                return (
                    f"基于最近 {tool} 成功结果收敛判断。{_switch_hints(tool, key_param)}"
                    " 若仍缺证据，提交 task.workbench 明确下一步可验证动作。"
                )
    return fallback


async def _record_continue_round_limit(
    *,
    loop: Any,
    ctx: Any,
    active_task: Any,
    tool_history: list[dict[str, Any]],
    max_inner_rounds: int,
) -> tuple[JudgmentOutput | None, Any | None]:
    if active_task is None or not registry_has_tool(loop._registry, "task.workbench"):
        return None, None
    recent_tools = _recent_tool_names(tool_history)
    next_verification = _specific_round_limit_next_verification(tool_history)
    action = build_workbench_action(
        workbench={
            "domain": "runtime-loop",
            "intent": "continue 阶段达到单 tick 工具续判上限，收敛到下一轮验证",
            "evidence": [
                f"本 tick continue 阶段已执行 {max_inner_rounds} 轮工具续判。",
                f"最近工具序列: {', '.join(recent_tools) if recent_tools else '（无）'}",
            ],
            "hypothesis": "当前任务仍需推进，但继续留在同一 tick 内追加工具会削弱总结与用户可见收敛。",
            "recovery_state": "continue_round_limit_reached",
            "next_verification": next_verification,
            "completion_checks": [
                "已停止在同一 tick 内继续追加工具调用。",
                "已把本轮工具结果收敛为下一轮的验证入口。",
            ],
        },
        rationale=(
            f"continue 阶段达到 {max_inner_rounds} 轮上限，先写入任务皮层收敛状态，"
            "避免单 tick 内无限续判。"
        ),
        next_step=next_verification,
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
    low_increment_budget = _continue_low_increment_budget(loop, max_inner_rounds)

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
            and registry_has_tool(loop._registry, "task.workbench")
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

        if (
            cont.decision == "act"
            and cont.chosen_action_id in _LOW_INCREMENT_CONTINUE_TOOLS
            and registry_has_tool(loop._registry, "task.workbench")
            and _low_increment_history_count(tool_history) >= low_increment_budget
        ):
            cont = _build_continue_low_increment_budget_action(
                action=cont,
                tool_name=cont.chosen_action_id,
                budget=low_increment_budget,
                history=tool_history,
            )

        if cont.decision == "act":
            _ensure_continue_task_id(cont, active_task, tool_history)
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
