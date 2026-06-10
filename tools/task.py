"""tools/task.py — 任务管理工具（供 LLM 通过判断层调用）。"""
from __future__ import annotations

import logging as _logging
import uuid
from typing import Any

from core.cortex import action_first_completion_blockers
from core.metabolic import (
    add_semantic_memory as metabolic_add_semantic_memory,
)
from core.metabolic import (
    amend_task as metabolic_amend_task,
)
from core.metabolic import (
    create_task as metabolic_create_task,
)
from core.metabolic import (
    mark_task_waiting as metabolic_mark_task_waiting,
)
from core.metabolic import (
    resume_task as metabolic_resume_task,
)
from core.metabolic import (
    update_task_data as metabolic_update_task_data,
)
from core.metabolic import (
    update_task_status as metabolic_update_task_status,
)
from store.task import build_task_similarity_query
from tools.registry import (
    CAPS_EXEMPT,
    ToolContext,
    ToolManifest,
    ToolParam,
    ToolResult,
    tool,
    tool_has_capability,
    tool_metadata,
)

_log_task_ops = _logging.getLogger("lingzhou.task_ops")


def _decision_basis(*parts: Any) -> str:
    """生成写入生命史账本的短依据摘要。"""
    text = " | ".join(str(part).strip() for part in parts if str(part or "").strip())
    return " ".join(text.split())[:240]


def _is_self_drive_growth_task(task: Any) -> bool:
    if getattr(task, "source", None) != "self_drive":
        return False
    result_json = getattr(task, "result_json", None)
    cortex = result_json.get("cortex") if isinstance(result_json, dict) else None
    return isinstance(cortex, dict) and str(cortex.get("intent") or "") == "self_drive_growth"


def _self_drive_growth_completion_blockers(task: Any, recent_runs: list[Any]) -> list[str]:
    if not _is_self_drive_growth_task(task):
        return []

    blockers: list[str] = []
    non_task_success = [
        run for run in recent_runs
        if str(getattr(run, "status", "") or "") == "succeeded"
        and str(getattr(run, "tool_name", "") or "")
        and not str(getattr(run, "tool_name", "") or "").startswith("task.")
    ]
    if not non_task_success:
        blockers.append("尚未执行非 task 工具取证，不能用“维持现状/成本高”直接完成自驱成长任务。")

    result_json = getattr(task, "result_json", None)
    cortex = result_json.get("cortex") if isinstance(result_json, dict) else {}
    evidence = cortex.get("evidence") if isinstance(cortex, dict) else None
    has_evidence = isinstance(evidence, list) and any(str(item or "").strip() for item in evidence)
    has_current_step = bool(str(getattr(task, "current_step", "") or "").strip())
    summary = str(result_json.get("summary") or "").strip() if isinstance(result_json, dict) else ""
    if not (has_evidence or has_current_step or summary):
        blockers.append("尚未把成长证据写入 task.workbench/current_step/result summary。")
    return blockers


_ACTIONABLE_VERIFICATION_MARKERS = (
    "定位",
    "查找",
    "读取",
    "检查",
    "验证",
    "确认",
    "测试",
    "运行",
    "修改",
    "修复",
    "实现",
    "改进",
    "编辑",
    "推送",
    "提交",
)
_NON_VERIFICATION_TOOLS = {
    "memory.add_semantic",
    "memory.add_wm",
    "memory.drop_wm",
    "task.workbench",
    "task.advance",
    "task.complete",
    "task.add",
    "task.update",
    "task.amend",
}
_VERIFICATION_TOOL_PREFIXES = (
    "file.",
    "shell.",
    "exec.",
    "process.",
    "probe.",
    "config.",
    "browser.",
    "web.",
    "memory.search",
    "memory.embed_backfill",
    "subagent.run",
)


def _task_cortex(task: Any) -> dict[str, Any]:
    result_json = getattr(task, "result_json", None)
    cortex = result_json.get("cortex") if isinstance(result_json, dict) else None
    return cortex if isinstance(cortex, dict) else {}


def _has_actionable_next_verification(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if any(marker in lowered for marker in ("无需", "不需要", "已完成", "完成该", "进入低频观察", "等待新用户")):
        return False
    return any(marker in value for marker in _ACTIONABLE_VERIFICATION_MARKERS)


def _is_successful_verification_run(run: Any) -> bool:
    if str(getattr(run, "status", "") or "").strip() != "succeeded":
        return False
    tool_name = str(getattr(run, "tool_name", "") or "").strip()
    if not tool_name or tool_name in _NON_VERIFICATION_TOOLS or tool_name.startswith("task."):
        return False
    return tool_name in _VERIFICATION_TOOL_PREFIXES or any(
        tool_name.startswith(prefix) for prefix in _VERIFICATION_TOOL_PREFIXES
    )


def _workbench_output_has_next_verification(run: Any) -> bool:
    output_json = getattr(run, "output_json", None)
    cortex = output_json.get("cortex") if isinstance(output_json, dict) else None
    return isinstance(cortex, dict) and _has_actionable_next_verification(str(cortex.get("next_verification") or ""))


def _unresolved_workbench_verification_blockers(task: Any, recent_runs: list[Any]) -> list[str]:
    cortex = _task_cortex(task)
    next_verification = str(cortex.get("next_verification") or "").strip()
    if not _has_actionable_next_verification(next_verification):
        return []

    latest_workbench_idx: int | None = None
    for idx, run in enumerate(recent_runs):
        if str(getattr(run, "tool_name", "") or "") == "task.workbench" and _workbench_output_has_next_verification(run):
            latest_workbench_idx = idx
            break

    if latest_workbench_idx is None:
        return []

    newer_runs = recent_runs[:latest_workbench_idx]
    if any(_is_successful_verification_run(run) for run in newer_runs):
        return []

    intent = str(cortex.get("intent") or "").strip()
    domain = str(cortex.get("domain") or "").strip()
    is_growth = _is_self_drive_growth_task(task) or intent == "self_drive_growth" or domain in {"self_evolution", "runtime", "memory_system"}
    if not is_growth:
        return []

    return [
        (
            "task.workbench 仍有未执行的 next_verification，不能把写 workbench/语义记忆当作完成："
            f"{next_verification[:180]}"
        )
    ]


async def _resolve_active_task(ctx: ToolContext):
    task = await ctx.get_active_task()
    if task is not None:
        return task
    task_store = getattr(ctx, "task_store", None)
    getter = getattr(task_store, "get_active", None)
    if getter is None:
        return None
    try:
        return await getter()
    except Exception:
        return None

async def _resolve_task(task_id: Any, ctx: ToolContext):
    """解析 task_id -> Task。None 时返回活跃任务；格式/查找错误时记录 warning 并回退。"""
    if task_id is None:
        return await _resolve_active_task(ctx)
    try:
        tid = int(task_id)
    except (ValueError, TypeError):
        _log_task_ops.warning("[task_ops] task_id=%r 格式无效（期望整数），回退到活跃任务", task_id)
        return await _resolve_active_task(ctx)
    try:
        return await ctx.task_store.get_task_by_id(tid)
    except Exception:
        _log_task_ops.warning("[task_ops] task_id=%d 不存在，回退到活跃任务", tid)
        return await _resolve_active_task(ctx)


def _task_metadata(
    task: Any,
    *,
    tool_name: str = "task",
    log_summary: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    summary = log_summary or f"{tool_name} task_id={task.id}"
    return tool_metadata(
        tool_name,
        summary,
        task_id=task.id,
        chain_id=task.chain_id,
        **extra,
    )


def _similar_task_source_filters(source: str) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    normalized_source = str(source or "").strip() or "external"
    if normalized_source == "self_drive":
        return ("self_drive",), None
    return None, ("self_drive",)


@tool(ToolManifest(
    name="task.advance",
    description="将活跃任务推进到 in_progress 状态并更新 next_step（首次取任务时调用）",
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
        params=[
        ToolParam("task_id", "number", "可选：显式指定要推进的任务 id；不传则使用当前 active task", required=False),
        ToolParam("next_step", "string", "计划的下一步描述", required=False),
    ],
))
async def task_advance(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可推进", skipped=True)
    if task.status == "in_progress":
        return ToolResult(summary=f"任务 [{task.id}] 已在进行中", skipped=True)
    if task.status == "done":
        return ToolResult(
            summary=f"任务 [{task.id}] 已完成，不能再次推进",
            skipped=True,
            resource_key=str(task.id),
            metadata=_task_metadata(task),
        )
    if task.status == "cancelled":
        return ToolResult(
            summary=f"任务 [{task.id}] 已取消，不能再次推进",
            skipped=True,
            resource_key=str(task.id),
            metadata=_task_metadata(task),
        )
    next_step = (params.get("next_step") or "").strip() or task.next_step
    await metabolic_update_task_status(
        ctx,
        task.id,
        status="in_progress",
        next_step=next_step,
        source="tools/task.advance",
        decision_basis=_decision_basis("advance task", task.title, next_step),
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已推进至 in_progress: {task.title}",
        evidence=f"task_id={task.id} next_step={next_step}",
        resource_key=str(task.id),
        state_delta={"task_status": "in_progress", "next_step": next_step},
        metadata=_task_metadata(
            task,
            tool_name="task.advance",
            log_summary=f"task.advance id={task.id} status=in_progress",
            next_step=next_step,
        ),
    )


@tool(ToolManifest(
    name="task.add",
    description="创建一个新任务",
    progress_category="mutation",
        params=[
        ToolParam("title", "string", "任务标题（简洁）", required=True),
        ToolParam("goal", "string", "任务目标（详细）", required=False),
        ToolParam("priority", "string", "优先级: low/normal/high/critical", required=False),
        ToolParam("source", "string", "可选：任务来源；默认 external，可显式设为 self_drive/curiosity/bootstrap 等", required=False),
        ToolParam("model_tier", "string", "可选：任务默认模型层级 reader/reasoner/repair", required=False),
        ToolParam("chain_id", "string", "可选：任务链 id；不传则自动继承或创建", required=False),
        ToolParam("parent_task_id", "number", "可选：父任务 id，用于形成任务链", required=False),
        ToolParam("current_step", "string", "可选：当前步骤名", required=False),
        ToolParam("next_step", "string", "可选：下一步说明", required=False),
    ],
))
async def task_add(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    title = (params.get("title") or "").strip()
    if not title:
        return ToolResult(summary="任务标题不能为空", skipped=True, error="ValidationError")
    goal = params.get("goal") or ""
    priority = params.get("priority") or "normal"
    source = (params.get("source") or "external").strip() or "external"
    parent_task_id = params.get("parent_task_id")
    parent = await _resolve_task(parent_task_id, ctx) if parent_task_id is not None else await ctx.get_active_task()
    chain_id = (params.get("chain_id") or (parent.chain_id if parent and parent.chain_id else f"chain-{uuid.uuid4().hex[:8]}"))
    current_step = (params.get("current_step") or "").strip()
    next_step = (params.get("next_step") or "").strip()
    model_tier = (params.get("model_tier") or "").strip()
    finder: Any = getattr(ctx.task_store, "find_similar_open_tasks", None)
    if finder is not None:
        allowed_sources, excluded_sources = _similar_task_source_filters(source)
        similar_tasks = await finder(
            build_task_similarity_query(title, goal, next_step),
            limit=1,
            min_score=ctx.config.thresholds.task_duplicate_reuse_score,
            allowed_sources=allowed_sources,
            excluded_sources=excluded_sources,
        )
        if similar_tasks:
            existing, score = similar_tasks[0]
            return ToolResult(
                summary=(
                    f"发现相似开放任务，复用已有任务: [{existing.id}] {existing.title} "
                    f"(status={existing.status}, score={score:.2f})"
                ),
                evidence=f"task_id={existing.id} similarity={score:.2f}",
                skipped=True,
                resource_key=str(existing.id),
                state_delta={
                    "task_status": existing.status,
                    "chain_id": existing.chain_id,
                    "source": existing.source,
                },
                metadata=_task_metadata(
                    existing,
                    tool_name="task.add",
                    log_summary=(
                        f"task.add reused id={existing.id} score={score:.2f}"
                    ),
                    parent_task_id=existing.parent_task_id,
                    source=existing.source,
                    reused_existing_task=True,
                    similarity_score=round(score, 3),
                ),
            )
    task_id = await metabolic_create_task(
        ctx,
        proposal_source="tools/task.add",
        decision_basis=_decision_basis("create task", title, goal, next_step),
        title=title,
        goal=goal,
        priority=priority,
        source=source,
        chain_id=chain_id,
        parent_task_id=str(parent.id) if parent else (str(parent_task_id or "") if parent_task_id else ""),
        current_step=current_step,
        next_step=next_step,
        model_tier=model_tier,
    )
    return ToolResult(
        summary=f"任务已创建: [{task_id}] {title}",
        evidence=f"task_id={task_id}",
        resource_key=str(task_id),
        state_delta={"task_status": "pending", "chain_id": chain_id, "source": source},
        metadata=tool_metadata(
            "task.add",
            f"task.add id={task_id} title={title[:80]!r}",
            task_id=task_id,
            chain_id=chain_id,
            parent_task_id=str(parent.id) if parent else "",
            source=source,
        ),
    )


@tool(ToolManifest(
    name="task.complete",
    description="将当前活跃任务标记为完成，并将任务叙事编译进语义记忆",
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
        params=[
        ToolParam("task_id", "number", "可选：显式指定要完成的任务 id；不传则使用当前 active task", required=False),
        ToolParam("force", "boolean", "可选：强制完成，跳过轻量证据门槛", required=False),
    ],
))
async def task_complete(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可完成", skipped=True)
    if task.status == "done":
        return ToolResult(
            summary=f"任务 [{task.id}] 已完成",
            skipped=True,
            resource_key=str(task.id),
            metadata=_task_metadata(task),
        )
    if task.status == "cancelled":
        return ToolResult(
            summary=f"任务 [{task.id}] 已取消，不能完成",
            skipped=True,
            resource_key=str(task.id),
            metadata=_task_metadata(task),
        )

    force = bool(params.get("force") or False)
    if not force and ctx.task_store is not None:
        # 自驱任务：若曾收到用户 inbox 消息，要求先为用户消息创建任务或回复，
        # 再完成自驱任务。确认已处理后可用 force=True 强制完成。
        if getattr(task, "source", None) == "self_drive" and isinstance(task.extras, dict) and task.extras.get("had_user_inbox"):
            return ToolResult(
                summary=(
                    f"任务 [{task.id}] 是自驱任务，但本轮收到过用户消息（inbox）。"
                    "请先用 task.add 为用户消息创建任务，或回复用户后再完成此任务。"
                    "若已处理，可传 force=True 强制完成。"
                ),
                error="UserInboxPending",
                skipped=True,
                metadata=_task_metadata(
                    task,
                    tool_name="task.complete",
                    log_summary=f"task.complete rejected UserInboxPending id={task.id}",
                ),
            )
        recent_runs = await ctx.task_store.list_runs(task_id=task.id, limit=12)
        self_drive_growth_blockers = _self_drive_growth_completion_blockers(task, recent_runs)
        if self_drive_growth_blockers:
            return ToolResult(
                summary=(
                    f"任务 [{task.id}] 暂不允许完成：自驱成长任务还没有形成最小成长证据。"
                    + " ".join(self_drive_growth_blockers)
                ),
                error="SelfDriveGrowthIncomplete",
                skipped=True,
                metadata=_task_metadata(
                    task,
                    tool_name="task.complete",
                    log_summary=f"task.complete rejected SelfDriveGrowthIncomplete id={task.id}",
                    blockers=self_drive_growth_blockers,
                ),
            )
        workbench_verification_blockers = _unresolved_workbench_verification_blockers(task, recent_runs)
        if workbench_verification_blockers:
            return ToolResult(
                summary=(
                    f"任务 [{task.id}] 暂不允许完成：任务皮层仍有未验证的下一步。"
                    + " ".join(workbench_verification_blockers)
                ),
                error="WorkbenchVerificationPending",
                skipped=True,
                metadata=_task_metadata(
                    task,
                    tool_name="task.complete",
                    log_summary=f"task.complete rejected WorkbenchVerificationPending id={task.id}",
                    blockers=workbench_verification_blockers,
                ),
            )
        action_first_blockers = action_first_completion_blockers(task=task, recent_runs=recent_runs)
        if action_first_blockers:
            return ToolResult(
                summary=(
                    f"任务 [{task.id}] 暂不允许完成：Action-first 执行任务仍缺少验收证据。"
                    + " ".join(action_first_blockers)
                ),
                error="ActionFirstCompletionBlocked",
                skipped=True,
                metadata=_task_metadata(
                    task,
                    tool_name="task.complete",
                    log_summary=f"task.complete rejected ActionFirstCompletionBlocked id={task.id}",
                    blockers=action_first_blockers,
                ),
            )
        recent_tools = [
            r.tool_name for r in recent_runs
            if r.status == "succeeded" and r.tool_name and not r.tool_name.startswith("task.")
        ]
        only_info_browsing = bool(recent_tools) and all(tool_has_capability(ctx.registry, t, "completion_info_only") for t in recent_tools)
        explicit_progress = bool((task.current_step or "").strip() or str(task.result_json.get("summary") or "").strip())
        if only_info_browsing and not explicit_progress:
            return ToolResult(
                summary=(
                    f"任务 [{task.id}] 暂不允许完成：最近仅有信息浏览类动作，"
                    "且缺少 current_step / result summary 等明确结论。"
                    "请先更新当前步骤或产出结论，再完成任务。"
                ),
                error="InsufficientEvidence",
                skipped=True,
                metadata=_task_metadata(
                    task,
                    tool_name="task.complete",
                    log_summary=f"task.complete rejected InsufficientEvidence id={task.id}",
                    recent_tools=recent_tools,
                ),
            )

        # 新门槛：如果最近有 mutation（manifest 标注 completion_mutation），
        # 要求其后至少有一个成功的验证动作（如跑测试）。
        # 注意：list_runs 返回的是 ORDER BY id DESC（最新在前），
        # 因此用 min(idx) 找最新 mutation，[:idx] 取比它更新的工具。
        has_mutation = any(tool_has_capability(ctx.registry, t, "completion_mutation") for t in recent_tools)
        if has_mutation:
            # 最新 mutation 的索引（DESC 顺序下 min = 最近）
            latest_mutation_idx = min(i for i, t in enumerate(recent_tools) if tool_has_capability(ctx.registry, t, "completion_mutation"))
            # 比最新 mutation 更近的工具（索引更小）
            post_mutation_tools = recent_tools[:latest_mutation_idx]
            # file.read 视为轻量验证（回读即确认）；shell.run 等 completion_verify 工具为强验证
            verified_after = any(
                tool_has_capability(ctx.registry, t, "completion_verify") or t == "file.read"
                for t in post_mutation_tools
            )
            if not verified_after:
                return ToolResult(
                    summary=(
                        f"任务 [{task.id}] 暂不允许完成：最近有修改动作"
                        f"（{recent_tools[latest_mutation_idx]}），"
                        "但缺少后续验证（如 shell.run 跑测试）。"
                        "请先验证修改正确性，再完成任务。"
                    ),
                    error="MutationWithoutVerification",
                    skipped=True,
                    metadata=_task_metadata(
                        task,
                        tool_name="task.complete",
                        log_summary=(
                            f"task.complete rejected MutationWithoutVerification id={task.id}"
                        ),
                        last_mutation=recent_tools[latest_mutation_idx],
                        post_mutation_tools=post_mutation_tools,
                    ),
                )

    await metabolic_update_task_status(
        ctx,
        task.id,
        status="done",
        next_step="completed via agent",
        source="tools/task.complete",
        decision_basis=_decision_basis("complete task", task.title, task.next_step),
    )

    # 自动 dismiss 该任务关联的未消除 failure（任务既然完成，旧失败已不再阻塞）
    task_failures = await ctx.task_store.list_failures_for_task(str(task.id), limit=30)
    dismissed_count = 0
    for f in task_failures:
        if not f.dismissed:
            await ctx.task_store.dismiss_failure(f.id)
            dismissed_count += 1

    # 程序性记忆编译：任务叙事 → 语义记忆节点（Anderson 1983 ACT-R）
    task_id_str = str(task.id)
    narrative = ctx.episodic.load_for_context(task_id_str, n_recent=5)
    if narrative.strip():
        node_id = f"skill-{uuid.uuid4().hex[:12]}"
        await metabolic_add_semantic_memory(
            ctx,
            node_id=node_id,
            kind="learned_skill",
            title=f"完成: task#{task.id} {task.title}",
            body=narrative,
            activation=0.8,
            valence=0.5,
            source="tools/task.complete",
            decision_basis=_decision_basis("compile completed task narrative", task.title),
        )
        return ToolResult(
            summary=f"任务 [{task.id}] 已完成，叙事已编译进语义记忆",
            evidence=f"task_id={task.id} skill_node={node_id}",
            resource_key=str(task.id),
            state_delta={"task_status": "done", "compiled_skill": node_id},
            metadata=_task_metadata(
                task,
                tool_name="task.complete",
                log_summary=f"task.complete id={task.id} compiled_skill={node_id}",
                skill_node=node_id,
            ),
        )

    return ToolResult(
        summary=f"任务 [{task.id}] 已完成",
        evidence=f"task_id={task.id}",
        resource_key=str(task.id),
        state_delta={"task_status": "done"},
        metadata=_task_metadata(
            task,
            tool_name="task.complete",
            log_summary=f"task.complete id={task.id}",
        ),
    )


@tool(ToolManifest(
    name="task.list",
    description="列出任务列表",
    prefer_tier="reader",
    progress_category="info",
    capabilities=("ask_evidence", *CAPS_EXEMPT, "completion_info_only"),
        params=[
        ToolParam("status", "string", "过滤状态: pending/in_progress/ready/resumed/waiting/done/all", required=False),
        ToolParam("limit", "number", "最多返回条数，默认 10", required=False),
    ],
))
async def task_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    status = params.get("status") or None
    if status == "all":
        status = None
    limit = int(params.get("limit") or 10)
    tasks = await ctx.task_store.list_tasks(status=status, limit=limit)
    if not tasks:
        return ToolResult(summary="没有匹配的任务", skipped=True)
    lines = []
    for t in tasks:
        chain = f" chain={t.chain_id}" if t.chain_id else ""
        wait = f" wait={t.wait_kind}:{t.wait_key}" if t.wait_kind else ""
        step = f" step={t.current_step}" if t.current_step else ""
        lines.append(f"[{t.id}] [{t.status}] [{t.priority}] {t.title}{chain}{step}{wait}")
    return ToolResult(
        summary="\n".join(lines),
        evidence=f"count={len(tasks)}",
        metadata=tool_metadata(
            "task.list",
            f"task.list count={len(tasks)} status={status or 'all'}",
            count=len(tasks),
        ),
    )


@tool(ToolManifest(
    name="task.update",
    description="更新当前活跃任务的 next_step 或状态。仅在有实质状态变更时调用，不用于记录思考进度。",
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
        params=[
        ToolParam("task_id", "number", "可选：显式指定要更新的任务 id；不传则使用当前 active task", required=False),
        ToolParam("next_step", "string", "下一步计划", required=False),
        ToolParam("status", "string", "新状态: pending/ready/in_progress/resumed/waiting/blocked/failed", required=False),
        ToolParam("current_step", "string", "当前步骤名", required=False),
        ToolParam("model_tier", "string", "可选：任务默认模型层级 reader/reasoner/repair", required=False),
    ],
))
async def task_update(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务", skipped=True)
    status = params.get("status") or task.status
    # 保护 waiting 状态：不允许用 task.update 将 waiting 降级，必须用 task.resume
    _downgrade_statuses = {"pending", "in_progress", "ready", "resumed"}
    if task.status == "waiting" and status in _downgrade_statuses:
        return ToolResult(
            summary=f"任务 [{task.id}] 当前处于 waiting，不能直接降级为 {status}，请使用 task.resume 恢复",
            skipped=True,
        )
    has_next_step = "next_step" in params
    has_current_step = "current_step" in params
    has_model_tier = "model_tier" in params
    next_step = str(params.get("next_step") or "").strip() if has_next_step else task.next_step
    current_step = str(params.get("current_step") or "").strip() if has_current_step else task.current_step
    model_tier = str(params.get("model_tier") or "").strip() if has_model_tier else task.model_tier
    await metabolic_update_task_status(
        ctx,
        task.id,
        status=status,
        next_step=next_step if has_next_step else None,
        current_step=current_step if has_current_step else None,
        model_tier=model_tier if has_model_tier else None,
        source="tools/task.update",
        decision_basis=_decision_basis("update task", task.title, status, current_step, next_step, model_tier),
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已更新: status={status}",
        evidence=f"task_id={task.id} next_step={next_step}",
        resource_key=str(task.id),
        state_delta={"task_status": status, "next_step": next_step, "current_step": current_step, "model_tier": model_tier},
        metadata=_task_metadata(
            task,
            tool_name="task.update",
            log_summary=f"task.update id={task.id} status={status}",
        ),
    )


@tool(ToolManifest(
    name="task.fail",
    description="将当前活跃任务标记为失败，记录失败原因并写入失败日志（触发进化反馈）",
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
        params=[
        ToolParam("task_id", "number", "可选：显式指定要失败的任务 id；不传则使用当前 active task", required=False),
        ToolParam("reason", "string", "失败原因摘要", required=True),
    ],
))
async def task_fail(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可标记失败", skipped=True)
    reason = (params.get("reason") or "未知原因").strip()
    await metabolic_update_task_status(
        ctx,
        task.id,
        status="failed",
        next_step=reason,
        source="tools/task.fail",
        decision_basis=_decision_basis("mark task failed", task.title, reason),
    )
    # 自动 dismiss 旧 failure，task 本身的 failed 状态就是最终记录
    task_failures = await ctx.task_store.list_failures_for_task(str(task.id), limit=30)
    for f in task_failures:
        if not f.dismissed:
            await ctx.task_store.dismiss_failure(f.id)
    await ctx.task_store.record_failure(
        kind="task_failure",
        summary=reason,
        context=f"task_id={task.id} title={task.title}",
        task_id=str(task.id),
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已标记失败: {reason}",
        evidence=f"task_id={task.id} reason={reason}",
        resource_key=str(task.id),
        state_delta={"task_status": "failed", "reason": reason},
        metadata=_task_metadata(
            task,
            tool_name="task.fail",
            log_summary=f"task.fail id={task.id}",
            reason=reason,
        ),
    )


@tool(ToolManifest(
    name="task.wait",
    description="把任务切到 waiting，并记录等待条件（外部结果 / 定时器 / 子任务等）",
    progress_category="mutation",
    capabilities=("completion_mutation", *CAPS_EXEMPT),
    params=[
        ToolParam("task_id", "number", "可选：显式指定任务 id；不传则使用当前 active task", required=False),
        ToolParam("wait_kind", "string", "等待类型，如 process/task/signal/time/external", required=True),
        ToolParam("wait_key", "string", "等待对象键，如 session_id / child_task_id / signal_key", required=False),
        ToolParam("current_step", "string", "当前步骤名", required=False),
        ToolParam("next_step", "string", "恢复后下一步", required=False),
    ],
))
async def task_wait(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="无活跃任务可等待", skipped=True)
    wait_kind = (params.get("wait_kind") or "").strip().lower()
    if not wait_kind:
        return ToolResult(summary="wait_kind 不能为空", skipped=True)
    wait_key = (params.get("wait_key") or "").strip()
    valid_wait_kinds = {"process", "task", "signal", "time", "external"}
    if wait_kind not in valid_wait_kinds:
        return ToolResult(summary=f"不支持的 wait_kind: {wait_kind}", skipped=True)
    current_step = str(params.get("current_step") or "").strip() if "current_step" in params else None
    next_step = str(params.get("next_step") or "").strip() if "next_step" in params else task.next_step
    await metabolic_mark_task_waiting(
        ctx,
        task.id,
        wait_kind=wait_kind,
        wait_key=wait_key,
        wait_json={"wait_kind": wait_kind, "wait_key": wait_key},
        current_step=current_step,
        next_step=next_step,
        source="tools/task.wait",
        decision_basis=_decision_basis("wait for dependency", task.title, wait_kind, wait_key, next_step),
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已进入 waiting: {wait_kind}{'/' + wait_key if wait_key else ''}",
        resource_key=str(task.id),
        state_delta={"task_status": "waiting", "wait_kind": wait_kind, "wait_key": wait_key},
        metadata=_task_metadata(
            task,
            tool_name="task.wait",
            log_summary=f"task.wait id={task.id} kind={wait_kind}",
            wait_kind=wait_kind,
            wait_key=wait_key,
        ),
    )


@tool(ToolManifest(
    name="task.resume",
    description="把 waiting/blocked 的任务恢复到 resumed/ready，并附带恢复结果",
    progress_category="mutation",
    capabilities=("completion_mutation", *CAPS_EXEMPT),
    params=[
        ToolParam("task_id", "number", "要恢复的任务 id", required=True),
        ToolParam("status", "string", "恢复后的状态，默认 resumed，也可设为 ready/in_progress", required=False),
        ToolParam("current_step", "string", "恢复后当前步骤名", required=False),
        ToolParam("next_step", "string", "恢复后下一步", required=False),
    ],
))
async def task_resume(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="找不到要恢复的任务", skipped=True)
    status = (params.get("status") or "resumed").strip()
    current_step = str(params.get("current_step") or "").strip() if "current_step" in params else None
    next_step = str(params.get("next_step") or "").strip() if "next_step" in params else task.next_step
    await metabolic_resume_task(
        ctx,
        task.id,
        status=status,
        current_step=current_step,
        next_step=next_step,
        result_json={"resumed_via": "task.resume"},
        source="tools/task.resume",
        decision_basis=_decision_basis("resume task", task.title, status, current_step, next_step),
    )
    return ToolResult(
        summary=f"任务 [{task.id}] 已恢复: status={status}",
        resource_key=str(task.id),
        state_delta={"task_status": status, "current_step": current_step if current_step is not None else task.current_step, "next_step": next_step if next_step is not None else task.next_step},
        metadata=_task_metadata(
            task,
            tool_name="task.resume",
            log_summary=f"task.resume id={task.id} status={status}",
            status=status,
        ),
    )


@tool(ToolManifest(
    name="task.steer",
    description="向指定任务的 inbox 注入转向指令；下一个 tick 该任务执行时将优先处理 inbox 消息",
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
    params=[
        ToolParam("task_id", "number", "目标任务 id；不传则使用当前 active task", required=False),
        ToolParam("message", "string", "转向指令内容（清晰描述新方向或修正要求）", required=True),
    ],
))
async def task_steer(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="找不到目标任务", skipped=True)
    message = (params.get("message") or "").strip()
    if not message:
        return ToolResult(summary="转向指令内容不能为空", skipped=True)
    existing: list = task.extras.get("inbox_messages") or []
    if not isinstance(existing, list):
        existing = []
    existing.append(message)
    await metabolic_update_task_data(
        ctx,
        task.id,
        {"inbox_messages": existing},
        source="tools/task.steer",
        decision_basis=_decision_basis("steer task", task.title, message),
    )
    return ToolResult(
        summary=f"已向任务 [{task.id}] 注入转向指令（inbox 共 {len(existing)} 条）",
        evidence=f"task_id={task.id} inbox_count={len(existing)}",
        resource_key=str(task.id),
        state_delta={"inbox_messages": len(existing)},
        metadata=_task_metadata(
            task,
            tool_name="task.steer",
            log_summary=f"task.steer id={task.id} inbox={len(existing)}",
            inbox_count=len(existing),
        ),
    )


@tool(ToolManifest(
    name="task.amend",
    description=(
        "修正任务的目标或标题。当新信息表明原始任务意图有误时使用（例如：补充说明让目标变了，"
        "之前的理解基于不完整信息）。与 task.steer 不同，amend 直接修改任务定义，不是注入转向指令。"
    ),
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
    params=[
        ToolParam("task_id", "number", "目标任务 id；不传则使用当前 active task", required=False),
        ToolParam("title", "string", "新的任务标题（不传则保持原标题）", required=False),
        ToolParam("goal", "string", "新的任务目标描述（不传则保持原目标）", required=False),
        ToolParam("priority", "string", "新的优先级 low/normal/high（不传则保持不变）", required=False),
        ToolParam("reason", "string", "修正原因（必填，说明为何需要纠正原意图）", required=True),
    ],
))
async def task_amend(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="找不到目标任务", skipped=True)
    reason = (params.get("reason") or "").strip()
    if not reason:
        return ToolResult(summary="必须提供修正原因 reason", skipped=True)
    new_title: str | None = (params.get("title") or "").strip() or None
    new_goal: str | None = params.get("goal")
    if new_goal is not None:
        new_goal = new_goal.strip() or None
    new_priority: str | None = (params.get("priority") or "").strip() or None

    if new_title is None and new_goal is None and new_priority is None:
        return ToolResult(summary="至少需要提供 title/goal/priority 之一", skipped=True)

    ok = await metabolic_amend_task(
        ctx,
        task.id,
        title=new_title,
        goal=new_goal,
        priority=new_priority,
        amendment_reason=reason,
        source="tools/task.amend",
        decision_basis=_decision_basis("amend task", task.title, reason, new_title, new_goal, new_priority),
    )
    if not ok:
        return ToolResult(summary=f"任务 [{task.id}] 修正失败（任务不存在？）", skipped=True)

    changed_parts = []
    if new_title:
        changed_parts.append(f"title={new_title!r}")
    if new_goal is not None:
        changed_parts.append(f"goal={new_goal[:60]!r}")
    if new_priority:
        changed_parts.append(f"priority={new_priority}")
    summary = f"任务 [{task.id}] 已修正：{', '.join(changed_parts)}"
    return ToolResult(
        summary=summary,
        evidence=f"task_id={task.id} reason={reason[:80]}",
        resource_key=str(task.id),
        state_delta={"amended": True},
        metadata=_task_metadata(
            task,
            tool_name="task.amend",
            log_summary=f"task.amend id={task.id} reason={reason[:60]}",
        ),
    )
