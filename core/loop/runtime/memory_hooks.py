"""core.loop.runtime.memory_hooks — 自驱/好奇心与记忆整合 helper。"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from core.metabolic import MetabolicEngine, StateProposal
from memory.consolidation import (
    build_consolidation_plan,
    build_daily_summary_node,
    current_week_key,
    merge_promoted_node,
)
from memory.working import WMItem

from ..cycle.focus import resolve_focus_task

_log = logging.getLogger("lingzhou.loop")


async def emit_self_drive_signal(loop: Any) -> None:
    """自驱力引擎：空闲或探索卡住时注入自主探索信号到 WM。"""
    # 只有 global:* 链才负责全局空转探索；chat/task 链有专职工作，不触发自驱
    chain_key = getattr(loop, "_current_chain_key", "")
    if chain_key and not chain_key.startswith("global:"):
        return

    # 跨链共享冷却（120s）：防止多个 global:* 链并发注入重复自驱 WM 信号
    now_mono = time.monotonic()
    if now_mono - loop._self_drive._last_injected_at < 120.0:
        return

    # 检查是否有真的活跃任务（非 waiting 状态）
    has_real_work = (
        loop._last_decision == "act"
        and loop._last_action_tool
        and not loop._last_action_tool.startswith("task.update")
    )
    # 补充检查：LLM 可能连续做 wait 决策等待子代理完成，此时 last_decision != "act"
    # 但 task store 里确实有活跃任务或运行中的 run，不应视为空闲
    if not has_real_work:
        active = await resolve_focus_task(loop)
        if active is not None:
            # source=self_drive 任务若 next_step 指向空转/监听，不视为真实工作，
            # 以防该任务一直挂着 in_progress 却不做事，导致自驱信号被永久压制。
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

    # 检查是否探索卡住（streak 超过窗口大小 + 2，使用公开属性）
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

    # 感知上下文：未完成 self_drive 任务数 + 上次完成时间，注入 WM 供 LLM 感知决策
    runnable = await loop._task_store.list_runnable_tasks(limit=20)
    pending_sd = [task for task in runnable if getattr(task, "source", None) == "self_drive"]
    recent_done = await loop._task_store.list_tasks(status="done", limit=10)
    last_done_ago = "无"
    for item in recent_done:
        if getattr(item, "source", None) != "self_drive":
            continue
        try:
            ts = datetime.fromisoformat(item.created_at.replace("Z", "+00:00")).timestamp()
            secs = int(time.time() - ts)
            if secs < 60:
                last_done_ago = f"{secs} 秒前"
            elif secs < 3600:
                last_done_ago = f"{secs // 60} 分钟前"
            elif secs < 86400:
                last_done_ago = f"{secs // 3600} 小时前"
            else:
                last_done_ago = f"{secs // 86400} 天前"
        except Exception:
            pass
        break

    task_template = loop._self_drive.generate_exploration_task(
        signal.suggested_domain or "self_evolution"
    )
    if signal.drive_type == "consolidate":
        drive_content = (
            f"[自驱信号·整合] 空闲 {loop._behavior.wait_streak} 轮，"
            f"自驱力 C={signal.curiosity_score:.2f}，模式=内聚整合。\n"
            f"触发原因: {signal.rationale}\n"
            f"待运行 self_drive 任务: {len(pending_sd)} 个；上次 self_drive 完成: {last_done_ago}\n"
            "本次请优先整合与巩固已有知识，而非开辟新方向：\n"
            "· 回顾最近几次任务的结论，写入语义记忆或情节记忆\n"
            "· 检查并更新 SOUL.md / DREAMS.md 的认知偏差\n"
            "· 检视近期失败，提取可复用的错误模式\n"
            "若认为当前状态仍需探索，可忽略此整合信号。"
        )
    else:
        drive_content = (
            f"[自驱信号] 空闲 {loop._behavior.wait_streak} 轮，"
            f"自驱力 C={signal.curiosity_score:.2f}。\n"
            f"触发原因: {signal.rationale}\n"
            f"建议方向: {signal.suggested_domain or 'self_evolution'}\n"
            f"待运行 self_drive 任务: {len(pending_sd)} 个；上次 self_drive 完成: {last_done_ago}\n"
            f"候选任务: {task_template['title']}\n"
            f"目标: {task_template['goal']}\n"
            f"下一步建议: {task_template.get('next_step', '(未提供)')}\n"
            "若认可这次自驱触发，可调用 task.add 创建任务；"
            "建议显式设置 source=self_drive，以便后续去重与追踪。\n"
            "本轮探索请优先读全相关文件（不加 limit），感知完整后再决定存储哪些结论。"
        )
    loop._wm.add(WMItem(
        kind="self_drive",
        content=drive_content,
        priority=loop._cfg.thresholds.wm_pri_signal,
    ))

    # 更新共享冷却时间戳，阻止其他 global 链在 120s 内重复注入
    loop._self_drive._last_injected_at = time.monotonic()
    # 自驱探索：强制下一 tick 使用 high thinking 以保障推理深度
    loop._pending_thinking_override = "high"

    _log.info(
        "[self_drive] 注入 WM 信号 C=%.2f domain=%s idle=%d rationale=%s",
        signal.curiosity_score,
        signal.suggested_domain,
        loop._behavior.wait_streak,
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
    pending_curiosity = [
        task for task in recent
        if getattr(task, "source", None) == "curiosity"
        and getattr(task, "status", "done") not in ("done", "failed")
    ]
    loop._last_curiosity_signal_idle_cycle = loop._idle_cycles
    _log.info(
        "[curiosity] idle=%d curiosity=%.2f pending_tasks=%d",
        loop._idle_cycles,
        curiosity,
        len(pending_curiosity),
    )

    # 无待处理的好奇心任务时，向 WM 注入信号，让 LLM 感知到好奇心触发并决定是否创建任务
    if not pending_curiosity:
        loop._wm.add(WMItem(
            kind="curiosity",
            content=(
                f"[好奇心] 已空闲 {loop._idle_cycles} 轮，好奇心 {curiosity:.2f} "
                f"> 阈值 {cfg.thresholds.curiosity_idle_task}。"
                "建议发起自主探索任务（task.add source=curiosity）或深化当前认知。"
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
        metabolic = getattr(loop, "_metabolic", None)
        if metabolic is None:
            metabolic = MetabolicEngine(loop._task_store)
        await metabolic.submit(StateProposal(
            op="set_fact",
            key=fact.key,
            value=fact.value,
            scope=fact.scope,
            source="loop/consolidation",
        ))

    for node in plan.semantic_nodes:
        merged = merge_promoted_node(loop._semantic.get(node.id), node, memory_cfg=loop._cfg.memory)
        loop._semantic.upsert(merged)

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
            loop._semantic.upsert(daily_summary_node)
        await (getattr(loop, "_metabolic", None) or MetabolicEngine(loop._task_store)).submit(StateProposal(
            op="set_fact",
            key=daily_summary_marker,
            value="1",
            scope="system",
            source="loop/daily_summary",
        ))

    # 保留身份锚点(bootstrap_identity)和自我感知信号(self_awareness)
    # self_awareness 包含行为循环检测等信号，清除后 LLM 会失去对空转的感知
    loop._wm.clear(preserve_kinds={"bootstrap_identity", "self_awareness"})

    # 清空后注入任务锚点,避免下一轮因 WM 为空而丢失任务上下文
    if active_task:
        progress_line = ""
        try:
            progress, progress_found = await loop._task_store.get_fact(f"task:{active_task.id}:progress")
            if progress_found and progress:
                progress_line = f"\n进度: {progress}"
        except Exception:
            pass
        loop._wm.add(WMItem(
            kind="task_anchor",
            content=(
                f"[任务锚点] {active_task.title}\n"
                f"目标: {active_task.goal or '(未指定)'}\n"
                f"下一步: {active_task.next_step or '(未指定)'}"
                f"{progress_line}"
            ),
            priority=0.95,
        ))

    # 同步感知基准,避免下一轮因 WM 大小骤降产生假预测误差
    loop._perception.reset_wm_baseline(len(loop._wm))
    _log.info(
        "[consolidate] WM items=%d semantic_promoted=%d facts_promoted=%d, WM cleared (bootstrap+task_anchor preserved)",
        len(items),
        len(plan.semantic_nodes),
        len(plan.facts),
    )
