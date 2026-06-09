"""core.loop.shared.continue_phase — tick 内续判工具循环。

从 tick.py 分离，避免 tick.py 超长。
由 tick() 在初始动作执行后、满足续判条件时调用。
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from memory.working import WMItem

from .common import (
    _maybe_reconcile_bootstrap,
    _next_initial_tier_hint,
    _resolve_thinking_override,
    _should_continue_within_tick,
    _tool_history_entry,
)
from .progress import action_key_param
from core.judgment.context.utils import _clip_for_context

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput

_log = logging.getLogger("lingzhou.loop")


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

    _inner = 0
    while True:
        if await loop._task_store.has_pending_chat_message():
            _log.debug("[continue] chat 消息到达，中断工具循环 inner=%d", _inner)
            break

        # 工具历史超长时压缩早期条目，避免上下文窗口爆炸
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

        if cont.decision == "act":
            tool_name = cont.chosen_action_id or ""
            key_param = action_key_param(cont.params)
            for behavior_item in loop._behavior.on_act(
                tool_name,
                key_param,
                str(active_task.id) if active_task else None,
                cont.params,
            ):
                loop._wm.add(behavior_item)
                _wm_delta.append(behavior_item.to_dict())
            loop._behavior.apply_cognitive_probe(cognitive_signals)
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
            if cont_result.error and "oldtextnotfound" in (cont_result.error or "").lower():
                for behavior_item in loop._behavior.on_edit_failure(cont_result.error or ""):
                    loop._wm.add(behavior_item)
            loop._behavior.on_act_result(cont.chosen_action_id or "", cont_result.summary or "")
            tool_history.append(_tool_history_entry(cont, cont_result))

        if cont.reply_to_user:
            cont.speech_intent = cont.reply_to_user
            cont.reply_to_user = ""

        action = cont
        result = cont_result
        _inner += 1
        if action.speech_intent or not _should_continue_within_tick(action, registry=loop._registry):
            break
        # PlanUnchanged：计划结构没变，继续循环只会死锁；跳出让下一 tick 直接执行具体工具。
        if cont_result.error == "PlanUnchanged":
            _log.debug("[continue] PlanUnchanged — 跳出 continue 循环，下一 tick 直接执行")
            break

    # ② continue 循环结束后同步检测（兜底 inner 轮删除 BOOTSTRAP.md 的场景）
    await _maybe_reconcile_bootstrap(loop)

    return action, result
