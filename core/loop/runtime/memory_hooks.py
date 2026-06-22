"""core.loop.runtime.memory_hooks — 自驱/好奇心与记忆整合 helper。"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from core.metabolic import add_semantic_memory, create_task, submit_fact
from core.judgment.tiers import REPLY_ONLY_FALLBACK_TIER
from memory.consolidation import (
    build_consolidation_plan,
    build_daily_summary_node,
    current_week_key,
    merge_promoted_node,
)
from memory.working import TASK_SWITCH_PRESERVE_KINDS, WMItem

from ..cycle.focus import claim_focus_task, resolve_focus_task

_log = logging.getLogger("lingzhou.loop")


def _describe_elapsed_seconds(seconds: float) -> str:
    """把时间差转成人类可读中文粗略区间，供自驱状态观察。"""
    secs = int(max(0, seconds))
    if secs < 60:
        return f"{secs} 秒前"
    if secs < 3600:
        return f"{secs // 60} 分钟前"
    if secs < 86400:
        return f"{secs // 3600} 小时前"
    return f"{secs // 86400} 天前"


def _fmt_drive_template_list(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return "- （未提供）"
    return "\n".join(f"- {str(item)}" for item in items)


def _self_drive_signal_lines(loop: Any, signal: Any, drive_type: str) -> list[str]:
    return [
        "[自驱事件]",
        f"type: {drive_type}",
        "scope: observation",
        f"idle_ticks: {loop._behavior.wait_streak}",
        f"curiosity_score: {signal.curiosity_score:.2f}",
        f"signal_rationale: {signal.rationale}",
    ]


def _self_drive_task_status_lines(
    open_count: int,
    created_task_id: int | None,
    last_done_ago: str,
) -> list[str]:
    return [
        f"open_self_drive_tasks: {open_count}",
        f"created_self_drive_task: {created_task_id or 'none'}",
        f"last_self_drive_done: {last_done_ago}",
    ]


async def _list_open_self_drive_tasks(loop: Any, *, limit: int = 20) -> list[Any]:
    task_store = getattr(loop, "_task_store", None)
    lister = getattr(task_store, "list_open_tasks", None)
    if callable(lister):
        tasks = await lister(limit=limit)
    else:
        tasks = await task_store.list_tasks(limit=limit)
    return [task for task in tasks if getattr(task, "source", None) == "self_drive"]


def build_task_anchor_item(
    active_task: Any,
    *,
    progress: str = "",
    action_feedback: str = "",
) -> WMItem:
    """构造统一任务锚点，供 tick 准备与 consolidate 复用。"""
    result_json = getattr(active_task, "result_json", {}) or {}
    cortex = result_json.get("cortex", {}) if isinstance(result_json, dict) else {}
    recovery_state = str((cortex.get("recovery_state") if isinstance(cortex, dict) else None) or "").strip()
    next_verification = str(
        (cortex.get("next_verification") if isinstance(cortex, dict) else None)
        or (cortex.get("next_experiment") if isinstance(cortex, dict) else None)
        or (cortex.get("verification") if isinstance(cortex, dict) else None)
        or ""
    ).strip()

    progress_line = f"\n进度: {progress}" if str(progress or "").strip() else ""
    feedback_line = f"\n上一动作反馈: {action_feedback}" if str(action_feedback or "").strip() else ""
    recovery_line = f"\n恢复状态: {recovery_state}" if recovery_state else ""
    verify_line = f"\n下一步验证: {next_verification}" if next_verification else ""
    return WMItem(
        kind="task_anchor",
        content=(
            f"[任务锚点] {getattr(active_task, 'title', '') or '(未命名任务)'}\n"
            f"目标: {getattr(active_task, 'goal', '') or '(未指定)'}\n"
            f"下一步: {getattr(active_task, 'next_step', '') or '(未指定)'}"
            f"{progress_line}"
            f"{feedback_line}"
            f"{recovery_line}"
            f"{verify_line}"
        ),
        priority=0.95,
    )


async def _create_self_drive_task(loop: Any, task_template: dict[str, Any], signal: Any) -> int | None:
    """把高置信自驱信号转成一个可推进的轻量任务，而不是只停留在 WM 念头。"""
    artifact = str(task_template.get("artifact") or "task.workbench 中包含 evidence、decision、next_step 的记录")
    evidence_needed = [
        str(item)
        for item in task_template.get("evidence_needed", [])
        if str(item or "").strip()
    ]
    cortex = {
        "domain": str(task_template.get("domain") or "self_evolution"),
        "intent": "self_drive_growth",
        "hypothesis": "当前空闲期可能存在一个低成本、可验证的自我成长机会。",
        "capabilities": ["读取状态/日志/任务", "整理记忆", "形成后续验证条件"],
        "evidence": [],
        "open_questions": [
            str(task_template.get("question") or "当前最值得验证的自驱成长问题是什么？"),
            "这个探索是否能改善连续性、能力边界或错误预防？",
        ],
        "next_verification": str(task_template.get("next_step") or "执行一次低成本取证动作。"),
        "completion_checks": [
            str(task_template.get("done_condition") or "能用具体证据回答问题，并写出下一步是否需要行动。"),
            f"产物已写入: {artifact}",
        ],
    }
    data = {
        "title": str(task_template.get("title") or "自驱成长探索"),
        "goal": str(task_template.get("goal") or ""),
        "priority": "low",
        "source": "self_drive",
        "status": "pending",
        "next_step": str(task_template.get("next_step") or ""),
        "model_tier": REPLY_ONLY_FALLBACK_TIER,
        "result_json": {"cortex": cortex},
        "extras": {
            "drive_type": str(getattr(signal, "drive_type", "") or "explore"),
            "curiosity_score": float(getattr(signal, "curiosity_score", 0.0) or 0.0),
            "rationale": str(getattr(signal, "rationale", "") or ""),
            "evidence_needed": evidence_needed,
            "artifact": artifact,
            "done_condition": str(task_template.get("done_condition") or ""),
        },
    }
    try:
        task_id = await create_task(
            loop,
            proposal_source="self_drive/auto_task",
            decision_basis="high-confidence self-drive signal with no pending self-drive task",
            **data,
        )
    except Exception:
        adder = getattr(getattr(loop, "_task_store", None), "add_task", None)
        if adder is None:
            return None
        task_id = await adder(**data)
    task = await loop._task_store.get_task_by_id(int(task_id))
    await claim_focus_task(loop, task, clear_current=True)
    return int(task_id)


async def emit_self_drive_signal(loop: Any) -> None:
    """将自驱意图转为可观察、可裁决的工作记忆事件。"""
    # 只有 global:* 链负责全局空转探索；chat/task 链有专职执行，不触发自驱事件。
    chain_key = getattr(loop, "_current_chain_key", "")
    if chain_key and not chain_key.startswith("global:"):
        return

    # 跨链共享冷却（120s）：防止多个 global:* 链重复注入同类自驱事件。
    now_mono = time.monotonic()
    if now_mono - loop._self_drive._last_injected_at < 120.0:
        return

    # 先判定当前是否已有真实工作，避免把忙碌状态误判为空闲并重复发信号。
    has_real_work = (
        loop._last_decision == "act"
        and loop._last_action_tool
        and not loop._last_action_tool.startswith("task.update")
    )
    # 补充检查：可能处于 wait/观察期，但 task_store 或 run 仍有进行中的状态。
    if not has_real_work:
        active = await resolve_focus_task(loop, include_waiting=True)
        if active is not None:
            # 自驱任务如果长期仅更新 next_step 但无实质变化，不应等同于真实推进工作。
            is_stalled_sd = (
                getattr(active, "source", None) == "self_drive"
                and loop._last_action_tool
                and loop._last_action_tool.startswith("task.update")
                and loop._last_decision == "act"
            )
            has_real_work = not is_stalled_sd
        else:
            running_runs = await loop._task_store.list_runs(status="running", limit=1)
            has_real_work = bool(running_runs)

    # 探索是否出现停滞（重复行为超过策略窗口）会提高自驱触发概率。
    stuck_gate = loop._cfg.loop.behavior_streak_threshold + 2
    explore_stuck = (
        loop._behavior.list_streak_count >= stuck_gate
        or loop._behavior.read_streak_count >= stuck_gate
    )

    signal = loop._self_drive.compute_signal(
        idle_ticks=loop._behavior.wait_streak,
        has_user_message=False,
        has_active_task=bool(has_real_work and not explore_stuck),
        tick=loop._judgment.self_model.tick_count,
        force_explore_idle=loop._cfg.thresholds.curiosity_idle_min_cycles,
    )
    if not signal.should_explore:
        return

    # 为 LLM 提供可裁决上下文：未完成自驱任务数 + 最近一次自驱完成时长。
    # waiting 也算 open；否则旧自驱任务一旦被挂起，空闲期会继续创建近似任务。
    open_sd = await _list_open_self_drive_tasks(loop, limit=20)
    if open_sd:
        _log.info(
            "[self_drive] skip signal open_self_drive_tasks=%d active_focus_preserved",
            len(open_sd),
        )
        return
    recent_done = await loop._task_store.list_tasks(status="done", limit=10)
    last_done_ago = "无"
    for item in recent_done:
        if getattr(item, "source", None) != "self_drive":
            continue
        try:
            ts = datetime.fromisoformat(item.created_at.replace("Z", "+00:00")).timestamp()
            last_done_ago = _describe_elapsed_seconds(time.time() - ts)
        except Exception:
            pass
        break

    task_template = loop._self_drive.generate_exploration_task(
        signal.suggested_domain or "self_evolution"
    )
    created_task_id = await _create_self_drive_task(loop, task_template, signal)
    status_lines = _self_drive_task_status_lines(len(open_sd), created_task_id, last_done_ago)
    if signal.drive_type == "consolidate":
        drive_content = "\n".join([
            *_self_drive_signal_lines(loop, signal, "consolidation"),
            *status_lines,
            "observed_need: recent traces may benefit from consolidation before new exploration.",
            "proposal:",
            "- consolidate_memory: 把近期自驱观察结果沉淀为可复用经验。",
            "- inspect_failures: 评估重复失败是否是可提炼的连续性边界。",
            "open_questions:",
            "- 哪些近期任务结论已经稳定，值得写入长期记忆？",
            "- 哪些失败或重复模式需要被提炼成可复用经验？",
            "- SOUL.md / DREAMS.md 是否出现与实际行为不一致的认知偏差？",
            "available_directions: consolidate_memory | inspect_failures | update_identity_reflection | ignore_signal",
        ])
    else:
        drive_content = "\n".join([
            *_self_drive_signal_lines(loop, signal, "exploration"),
            f"candidate_domain: {signal.suggested_domain or 'self_evolution'}",
            *status_lines,
            f"candidate_task_title: {task_template['title']}",
            f"candidate_task_goal: {task_template['goal']}",
            f"candidate_next_step: {task_template.get('next_step', '(未提供)')}",
            f"candidate_question: {task_template.get('question', '(未提供)')}",
            "candidate_evidence_needed:",
            _fmt_drive_template_list(task_template.get('evidence_needed')),
            f"candidate_artifact: {task_template.get('artifact', '(未提供)')}",
            f"candidate_done_condition: {task_template.get('done_condition', '(未提供)')}",
            "proposal:",
            "- create_task: 为候选方向建立一次轻量探索任务。",
            "- observe_more: 先补证据再决策。",
            "- consolidate_first: 先完成 consolidation，再重评。",
            "open_questions:",
            "- 这个候选方向是否真的能改善当前生命连续性或能力边界？",
            "- 是否已有未完成 self_drive 任务覆盖同一问题？",
            "- 当前证据是否足够创建任务，还是应先观察、等待或忽略？",
            "available_directions: create_self_drive_task | gather_evidence | consolidate_first | ignore_signal",
        ])
    loop._wm.add(WMItem(
        kind="self_drive",
        content=drive_content,
        priority=loop._cfg.thresholds.wm_pri_signal,
    ))

    # 更新共享冷却时间戳，避免 global 链重复注入同类事件。
    loop._self_drive._last_injected_at = time.monotonic()

    _log.info(
        "[self_drive] 注入 WM 信号 C=%.2f domain=%s idle=%d created_task=%s rationale=%s",
        signal.curiosity_score,
        signal.suggested_domain,
        loop._behavior.wait_streak,
        created_task_id or "-",
        signal.rationale,
    )


async def emit_curiosity_signal(loop: Any, ethos_state: Any) -> None:
    """P1-C: 好奇心阈值驱动的探索信号注入。"""
    cfg = loop._cfg
    if loop._idle_cycles < cfg.thresholds.curiosity_idle_min_cycles:
        return

    curiosity = getattr(ethos_state.values, "curiosity", 0.0) if ethos_state else 0.0
    if curiosity < cfg.thresholds.curiosity_idle_task:
        return

    if loop._idle_cycles - loop._last_curiosity_signal_idle_cycle < cfg.thresholds.curiosity_idle_min_cycles:
        return

    recent = await loop._task_store.list_tasks(limit=10)
    open_tasks = [
        task for task in recent
        if getattr(task, "status", "done") not in ("done", "failed")
    ]
    pending_curiosity = [
        task for task in open_tasks
        if getattr(task, "source", None) == "curiosity"
    ]
    if open_tasks:
        _log.info(
            "[curiosity] skip open_tasks=%d pending_curiosity_tasks=%d",
            len(open_tasks),
            len(pending_curiosity),
        )
        return
    loop._last_curiosity_signal_idle_cycle = loop._idle_cycles
    _log.info(
        "[curiosity] idle=%d curiosity=%.2f pending_tasks=%d",
        loop._idle_cycles,
        curiosity,
        len(pending_curiosity),
    )

    # 无待处理的好奇心任务时，向 WM 注入事件，让 LLM 感知触发原因并自行裁决
    if not pending_curiosity:
        loop._wm.add(WMItem(
            kind="curiosity",
            content=(
                "[好奇心事件]\n"
                f"idle_cycles: {loop._idle_cycles}\n"
                f"curiosity: {curiosity:.2f}\n"
                f"threshold: {cfg.thresholds.curiosity_idle_task}\n"
                f"pending_curiosity_tasks: {len(pending_curiosity)}\n"
                "observation: curiosity is above threshold while no curiosity task is pending.\n"
                "open_questions:\n"
                "- 当前是否存在值得主动探索或深化的未解问题？\n"
                "- 是否应保持安静，等待外部输入或更多证据？\n"
                "available_directions: create_curiosity_task | deepen_current_cognition | wait"
            ),
            priority=0.7,
        ))


async def consolidate(loop: Any, active_task: Any) -> None:
    """将 WM 分流到情节记忆、长期语义层和 durable facts。"""
    items = loop._wm.get_top(25)
    if not items:
        return

    task_id = str(active_task.id) if active_task else None
    plan = build_consolidation_plan(
        items,
        task_id=task_id,
        task_title=active_task.title if active_task else None,
        memory_cfg=loop._cfg.memory,
        emotion_valence=loop._emotion.valence,
    )
    if plan.episodic_summary:
        loop._episodic.record(role="consolidation", content=plan.episodic_summary, task_id=task_id)

    for fact in plan.facts:
        await submit_fact(
            loop,
            key=fact.key,
            value=fact.value,
            scope=fact.scope,
            source="loop/consolidation",
        )

    for node in plan.semantic_nodes:
        merged = merge_promoted_node(loop._semantic.get(node.id), node, memory_cfg=loop._cfg.memory)
        await add_semantic_memory(
            loop,
            node_id=merged.id,
            kind=merged.kind,
            title=merged.title,
            body=merged.body,
            activation=merged.activation,
            valence=merged.valence,
            importance=getattr(merged, "importance", 0.0),
            tags=merged.tags or [],
            created_at=merged.created_at,
            source="loop/consolidation",
            decision_basis=f"consolidation_plan_node:{node.id}",
        )

    today_stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    week_key = current_week_key()
    daily_summary_marker = f"memory:daily_summary:{week_key}:{today_stamp}"
    _, daily_summary_done = await loop._task_store.get_fact(daily_summary_marker)
    if not daily_summary_done:
        recent_daily_text = loop._episodic.load_recent_daily_context(
            loop._cfg.memory.daily_summary_days,
            loop._cfg.memory.daily_summary_max_chars,
        )
        daily_summary_node = build_daily_summary_node(
            recent_daily_text,
            week_key=week_key,
            memory_cfg=loop._cfg.memory,
            emotion_valence=loop._emotion.valence,
            existing=loop._semantic.get(f"daily-summary-{week_key}"),
        )
        if daily_summary_node is not None:
            await add_semantic_memory(
                loop,
                node_id=daily_summary_node.id,
                kind=daily_summary_node.kind,
                title=daily_summary_node.title,
                body=daily_summary_node.body,
                activation=daily_summary_node.activation,
                valence=daily_summary_node.valence,
                importance=daily_summary_node.importance,
                tags=daily_summary_node.tags or [],
                created_at=daily_summary_node.created_at,
                source="loop/daily_summary",
                decision_basis="consolidation_daily_summary",
            )
        await submit_fact(
            loop,
            key=daily_summary_marker,
            value="1",
            scope="system",
            source="loop/daily_summary",
        )

    # 保留身份锚点、任务关键语境与自我感知信号，避免整合后失去任务切换所需上下文
    loop._wm.clear(preserve_kinds=set(TASK_SWITCH_PRESERVE_KINDS))

    # 清空后注入任务锚点,避免下一轮因 WM 为空而丢失任务上下文
    if active_task:
        progress_line = ""
        try:
            progress, progress_found = await loop._task_store.get_fact(f"task:{active_task.id}:progress")
            if progress_found and progress:
                progress_line = str(progress)
        except Exception:
            pass
        loop._wm.add(build_task_anchor_item(active_task, progress=progress_line))

    # 同步感知基准,避免下一轮因 WM 大小骤降产生假预测误差
    loop._perception.reset_wm_baseline(len(loop._wm))
    preserved_kinds = ",".join(sorted(TASK_SWITCH_PRESERVE_KINDS))
    _log.info(
        "[consolidate] WM items=%d semantic_promoted=%d facts_promoted=%d, WM preserved kinds=%s",
        len(items),
        len(plan.semantic_nodes),
        len(plan.facts),
        preserved_kinds,
    )
