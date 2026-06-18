"""core.loop.task.runtime — 任务级 meta-reflection 摄入、运行时 hint 与进度同步。"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from core.cortex import intent as cortex_intent
from core.judgment.tiers import JUDGMENT_TIERS, is_judgment_tier
from core.metabolic import submit_fact, update_task_data
from memory.working import WMItem, WorkingMemory

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from store.task import Task, TaskStore

_log = logging.getLogger("lingzhou.loop")
_RUNTIME_TIER_HINT_SCAN_ORDER = tuple(reversed(JUDGMENT_TIERS))

def _suggest_tier_from_text(text: str) -> str | None:
    lowered = (text or "").lower()
    for tier in _RUNTIME_TIER_HINT_SCAN_ORDER:
        if tier in lowered:
            return tier
    return None


def _meta_reflection_set_fact_instruction(key: str, value: Any, *, scope: str = "system") -> str:
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (
        "该建议尚未自动生效。若认可，请调用 memory.set_fact，"
        f"key={key}，scope={scope}，value={serialized}。"
    )


def _normalize_routing_tier(text: str) -> str:
    """从反思建议文本中抽取可落库的 routing tier。"""
    return _suggest_tier_from_text(text) or ""


def _loads_json_dict(raw: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(str(raw or ""))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


async def _submit_json_fact(
    fact_owner: Any,
    *,
    key: str,
    value: dict[str, Any],
    scope: str,
    source: str = "task_runtime/ingest",
) -> None:
    await submit_fact(
        fact_owner,
        key=key,
        value=json.dumps(value, ensure_ascii=False),
        scope=scope,
        source=source,
    )


def _reflection_payload(reflection: Any, **extra: Any) -> dict[str, Any]:
    payload = {
        "reflection_id": reflection.id,
        "decision": reflection.decision,
        "proposal": reflection.proposal,
        "verification_plan": reflection.verification_plan,
    }
    payload.update(extra)
    return payload


async def _ingest_actionable_meta_reflections(task_store: TaskStore, wm: WorkingMemory, metabolic: Any | None = None) -> list[str]:
    injected: list[str] = []
    fact_owner = metabolic or task_store
    for reflection in await task_store.list_meta_reflections(limit=10):
        if reflection.decision not in {"apply", "rollback"}:
            continue
        fact_key = f"meta_reflection:ingested:{reflection.id}"
        _, found = await task_store.get_fact(fact_key)
        if found:
            continue
        applied_change = "recorded"
        followup = ""
        if reflection.target_kind == "threshold":
            raw_policy, policy_found = await task_store.get_fact("control:durable_failure_policy")
            policy = {"threshold": 3, "ttl_sec": 7200}
            if policy_found and raw_policy.strip():
                loaded = _loads_json_dict(raw_policy)
                policy["threshold"] = int(loaded.get("threshold") or policy["threshold"])
                policy["ttl_sec"] = int(loaded.get("ttl_sec") or policy["ttl_sec"])
            if reflection.decision == "rollback":
                policy = {"threshold": 3, "ttl_sec": 7200}
                applied_change = "queued durable failure policy rollback hint"
            else:
                policy["threshold"] = max(1, policy["threshold"] + 1)
                policy["ttl_sec"] = max(900, policy["ttl_sec"] // 2)
                applied_change = (
                    f"queued durable failure policy hint threshold={policy['threshold']} ttl={policy['ttl_sec']}"
                )
            await _submit_json_fact(
                fact_owner,
                key="control:meta_reflection_hint:threshold",
                value=_reflection_payload(reflection, suggested_policy=policy),
                scope="system",
            )
            followup = _meta_reflection_set_fact_instruction(
                "control:durable_failure_policy",
                policy,
                scope="system",
            )
        elif reflection.target_kind == "task_split" and reflection.task_id:
            await _submit_json_fact(
                fact_owner,
                key=f"task:{reflection.task_id}:needs_replan",
                value=_reflection_payload(reflection),
                scope="task",
            )
            applied_change = "set task replan hint"
            followup = "该建议尚未自动写回任务。若认可，请调用 task.update 修改 next_step。"
        elif reflection.target_kind == "routing":
            preferred_tier = _normalize_routing_tier(reflection.proposal)
            if reflection.task_id:
                await _submit_json_fact(
                    fact_owner,
                    key=f"task:{reflection.task_id}:routing_guard",
                    value=_reflection_payload(
                        reflection,
                        tool_name=reflection.tool_name,
                        preferred_tier=preferred_tier,
                    ),
                    scope="task",
                )
            await _submit_json_fact(
                fact_owner,
                key="control:meta_reflection_hint:routing",
                value=_reflection_payload(
                    reflection,
                    tool_name=reflection.tool_name,
                    preferred_tier=preferred_tier,
                    task_id=reflection.task_id,
                ),
                scope="system",
            )
            if reflection.task_id:
                applied_change = "queued task routing guard"
                if is_judgment_tier(preferred_tier):
                    followup = f"该建议已识别 routing tier={preferred_tier}，已进入下一轮任务提示闭环"
                else:
                    followup = (
                        "该建议未识别到明确 tier。若认可，请调用 task.update 修改 model_tier，"
                        "或补充 routing 反思建议中的层级词。"
                    )
            elif reflection.decision == "rollback":
                applied_change = "queued routing rollback hint"
                followup = _meta_reflection_set_fact_instruction("pref:routing_overrides", "", scope="system")
            else:
                applied_change = "queued routing control hint"
                followup = "该建议尚未自动改写全局路由。若认可，请调用 memory.set_fact 更新 pref:routing_overrides。"
        else:
            await _submit_json_fact(
                fact_owner,
                key=f"control:meta_reflection_hint:{reflection.target_kind}",
                value=_reflection_payload(reflection),
                scope="system",
            )
            applied_change = f"queued {reflection.target_kind} control hint"
            followup = "该建议尚未自动生效。若认可，请调用 memory.set_fact 写入相应 control/pref 事实。"
        wm.add(WMItem(
            kind="meta_reflection",
            content=(
                f"[双环反思 {reflection.decision}] target={reflection.target_kind} tool={reflection.tool_name or 'unknown'}\n"
                f"已处理：{applied_change}\n"
                f"诊断：{reflection.diagnosis}\n"
                f"建议：{reflection.proposal}\n"
                f"验证：{reflection.verification_plan}"
                + (f"\n处理建议：{followup}" if followup else "")
            ),
            priority=0.76 if reflection.decision == "rollback" else 0.72,
        ))
        if reflection.task_id:
            await _submit_json_fact(
                fact_owner,
                key=f"task:{reflection.task_id}:meta_reflection",
                value=_reflection_payload(
                    reflection,
                    target_kind=reflection.target_kind,
                ),
                scope="task",
            )
        await submit_fact(
            fact_owner,
            key=fact_key,
            value=datetime.now(UTC).isoformat(),
            scope="system",
            source="task_runtime/ingest",
        )
        _log.info("[meta-reflection] surfaced reflection=%s target=%s change=%s", reflection.id, reflection.target_kind, applied_change)
        injected.append(reflection.id)
    return injected


async def _consume_task_runtime_hints(
    task_store: TaskStore,
    task: Task | None,
    wm: WorkingMemory,
    metabolic: Any | None = None,
) -> Task | None:
    if task is None:
        return None
    if metabolic is None:
        metabolic = task_store

    last_replan_id = str(task.extras.get("last_replan_reflection_id") or "")
    raw_replan, replan_found = await task_store.get_fact(f"task:{task.id}:needs_replan")
    if replan_found and raw_replan.strip():
        replan = _loads_json_dict(raw_replan)
        reflection_id = str(replan.get("reflection_id") or "")
        if reflection_id and reflection_id != last_replan_id:
            proposal = str(replan.get("proposal") or "").strip()
            verification = str(replan.get("verification_plan") or "").strip()
            replan_step = proposal or verification or "先重拆任务，再继续执行。"
            await update_task_data(
                metabolic,
                task.id,
                {"last_replan_reflection_id": reflection_id},
                source="task_runtime/hint",
            )
            task.extras["last_replan_reflection_id"] = reflection_id
            wm.add(WMItem(
                kind="task_replan",
                content=(
                    f"[任务重规划建议] task#{task.id}\n"
                    f"建议 next_step: {replan_step}\n"
                    f"验证: {verification or '（无）'}\n"
                    "该建议尚未自动写回任务。若认可，请调用 task.update 修改 next_step。"
                ),
                priority=0.84,
            ))
            _log.info("[runtime-hint] task=%s surfaced replan hint=%s", task.id, replan_step)

    last_meta_id = str(task.extras.get("last_task_meta_reflection_id") or "")
    raw_meta, meta_found = await task_store.get_fact(f"task:{task.id}:meta_reflection")
    if meta_found and raw_meta.strip():
        meta_payload = _loads_json_dict(raw_meta)
        reflection_id = str(meta_payload.get("reflection_id") or "")
        if reflection_id and reflection_id != last_meta_id:
            target_kind = str(meta_payload.get("target_kind") or "reflection")
            decision = str(meta_payload.get("decision") or "defer")
            proposal = str(meta_payload.get("proposal") or "").strip()
            verification = str(meta_payload.get("verification_plan") or "").strip()
            wm.add(WMItem(
                kind="task_reflection",
                content=(
                    f"[任务级反思 {decision}] target={target_kind}\n"
                    f"建议：{proposal or '（无）'}\n"
                    f"验证：{verification or '（无）'}"
                ),
                priority=0.78,
            ))
            await update_task_data(
                metabolic,
                task.id,
                {"last_task_meta_reflection_id": reflection_id},
                source="task_runtime/hint",
            )
            task.extras["last_task_meta_reflection_id"] = reflection_id
            _log.info("[runtime-hint] task=%s surface task meta reflection=%s", task.id, reflection_id)

    last_routing_id = str(task.extras.get("last_routing_reflection_id") or "")
    raw_guard, guard_found = await task_store.get_fact(f"task:{task.id}:routing_guard")
    if guard_found and raw_guard.strip():
        guard = _loads_json_dict(raw_guard)
        reflection_id = str(guard.get("reflection_id") or "")
        if reflection_id and reflection_id != last_routing_id:
            tool_name = str(guard.get("tool_name") or "unknown")
            proposal = str(guard.get("proposal") or "").strip()
            decision = str(guard.get("decision") or "apply").strip()
            preferred_tier = _normalize_routing_tier(proposal)
            tier = preferred_tier if is_judgment_tier(preferred_tier) else ""
            update_payload: dict[str, Any] = {"last_routing_reflection_id": reflection_id}
            auto_applied_tier = False
            if tier:
                update_payload["model_tier"] = tier
                auto_applied_tier = task.model_tier != tier
            if decision == "rollback" and "model_tier" not in update_payload and task.model_tier:
                update_payload["model_tier"] = ""
            await update_task_data(
                metabolic,
                task.id,
                update_payload,
                source="task_runtime/hint",
            )
            task.extras["last_routing_reflection_id"] = reflection_id
            if tier:
                task.model_tier = tier
            elif decision == "rollback" and "model_tier" in update_payload:
                task.model_tier = ""
            wm.add(WMItem(
                kind="routing_guard",
                content=(
                    f"[路由护栏建议] task#{task.id} tool={tool_name}\n"
                    f"建议 tier: {tier or '未识别'}\n"
                    f"理由: {proposal or '基于任务路由建议更新 model_tier。'}\n"
                    + (
                        f"已自动写回 task.model_tier={tier}。"
                        if tier and auto_applied_tier
                        else "该建议已写回当前任务状态（如与现状一致则不发生变化）。" if tier
                        else "未识别到可落库 tier，未自动改写 model_tier。"
                    )
                ),
                priority=0.82,
            ))
            _log.info(
                "[runtime-hint] task=%s surfaced routing guard tier=%s applied=%s decision=%s tool=%s",
                task.id,
                tier or "unparsed",
                auto_applied_tier,
                decision,
                tool_name,
            )

    return task


async def _sync_task_progress_state(
    task_store: TaskStore,
    task: Task | None,
    *,
    previous_next_step: str,
    action: JudgmentOutput,
    progressful: bool,
    state_delta: dict[str, Any] | None = None,
) -> Task | None:
    if task is None:
        return None

    latest = await task_store.get_task_by_id(task.id) or task
    planned_next = str(action.next_step or "").strip()
    state_next = ""
    state_next_authoritative = False
    if isinstance(state_delta, dict):
        cortex_delta = state_delta.get("cortex")
        cortex_next = ""
        if isinstance(cortex_delta, dict):
            cortex_next = str(
                cortex_delta.get("next_step")
                or cortex_delta.get("recovery_next_step")
                or cortex_delta.get("next_verification")
                or ""
            ).strip()
        state_next = str(
            state_delta.get("next_step")
            or state_delta.get("recovery_next_step")
            or state_delta.get("next_verification")
            or cortex_next
            or ""
        ).strip()
        state_next = cortex_intent.clean_next_verification_text(state_next)
        action_tool = str(getattr(action, "chosen_action_id", "") or "").strip()
        state_next_authoritative = bool(
            state_next
            and (
                action_tool == "task.workbench"
                or state_delta.get("completion_blocked") is True
                or state_delta.get("tool_input_invalid") is True
            )
        )
    explicit_current_step = None
    if state_delta is not None and "current_step" in state_delta:
        explicit_current_step = str(state_delta.get("current_step") or "").strip()
    current_step = latest.current_step
    next_step = latest.next_step
    updated = False

    if explicit_current_step is not None and current_step != explicit_current_step:
        current_step = explicit_current_step
        updated = True

    if progressful and previous_next_step:
        if explicit_current_step is None and current_step != previous_next_step:
            current_step = previous_next_step
            updated = True
        if planned_next:
            if not next_step or next_step == previous_next_step:
                next_step = planned_next
                updated = True
        elif state_next:
            if not next_step or next_step == previous_next_step or state_next_authoritative:
                next_step = state_next
                updated = True
        elif next_step == previous_next_step:
            next_step = ""
            updated = True
    elif planned_next and not next_step:
        next_step = planned_next
        updated = True
    elif state_next and (not next_step or state_next_authoritative):
        next_step = state_next
        updated = True

    if not updated:
        return latest

    await task_store.sync_task_progress(
        latest.id,
        current_step=current_step,
        next_step=next_step,
    )
    _log.info(
        "[task-progress] task=%s current_step=%s next_step=%s progressful=%s",
        latest.id,
        current_step,
        next_step,
        progressful,
    )
    refreshed = await task_store.get_task_by_id(latest.id)
    return refreshed or latest
