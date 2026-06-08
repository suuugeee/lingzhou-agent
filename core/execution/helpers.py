"""core/execution/helpers.py — 执行层 helper（finalize、run 记忆、降噪）。"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from core.config_models import ThresholdsConfig, run_result_memory_affect
from core.contracts.execution import action_key_param
from core.log_fields import execution_scope_fields
from core.metabolic import add_semantic_memory, update_run, update_task_result
from store.task import build_task_run_result_patch
from tools.registry import ToolContext, ToolResult, tool_has_capability

_log = logging.getLogger("lingzhou.execution")

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentOutput
    from tools.registry import ToolRegistry
    from tools.view_protocols import TaskStoreViewProtocol


_THRESHOLDS_DEFAULTS = ThresholdsConfig()
_RUN_OUTCOME_MEMORY_FIELD_MAX_CHARS = 6000


def _default_durable_failure_policy() -> dict[str, int]:
    return {
        "threshold": _THRESHOLDS_DEFAULTS.durable_failure_threshold,
        "ttl_sec": _THRESHOLDS_DEFAULTS.durable_failure_ttl_sec,
    }


def _clip_run_outcome_memory_field(text: str, *, limit: int = _RUN_OUTCOME_MEMORY_FIELD_MAX_CHARS) -> str:
    """压缩 run_result 语义记忆字段，避免大 stdout 污染检索路径。"""
    value = str(text or "")
    if len(value) <= limit:
        return value
    keep_each_side = max(200, (limit - 120) // 2)
    omitted = len(value) - keep_each_side * 2
    return (
        value[:keep_each_side]
        + f"\n...[run_result memory truncated, omitted {omitted} chars]...\n"
        + value[-keep_each_side:]
    )


async def _load_durable_failure_policy(task_store: TaskStoreViewProtocol | None) -> dict[str, int]:
    policy = _default_durable_failure_policy()
    if task_store is None:
        return policy
    raw, found = await task_store.get_fact("control:durable_failure_policy")
    if not found or not raw.strip():
        return policy
    try:
        data = json.loads(raw)
    except Exception:
        return policy
    threshold = int(data.get("threshold") or policy["threshold"])
    ttl_sec = int(data.get("ttl_sec") or policy["ttl_sec"])
    if threshold > 0:
        policy["threshold"] = threshold
    if ttl_sec > 0:
        policy["ttl_sec"] = ttl_sec
    return policy


def _failure_fact_key(action: JudgmentOutput) -> str:
    sig = f"{action.chosen_action_id or ''}|{action_key_param(action.params)}"
    digest = hashlib.md5(sig.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"durable_failure:{digest}"


def _coerce_result_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    return str(value)


def _classify_durable_failure(result: ToolResult) -> str | None:
    # 进化修复：防御上游误传 dict 导致 TypeError
    if isinstance(result, dict):
        return None
    if not hasattr(result, "summary"):
        return None
    text = "\n".join(
        segment
        for segment in (
            _coerce_result_text(result.summary),
            _coerce_result_text(result.error or ""),
            _coerce_result_text(result.evidence),
        )
        if segment
    ).lower()
    patterns = {
        "missing_path": [
            "no such file or directory", "路径不存在", "文件不存在", "未找到", "找不到脚本",
        ],
        "not_a_directory": ["not a directory", "不是目录"],
        "not_a_file": ["not a file", "不是文件"],
        "empty_path": ["path 不能为空", "emptypath"],
        "command_not_found": ["command not found", "工具不存在"],
    }
    for code, needles in patterns.items():
        if any(n in text for n in needles):
            return code
    return None


def _normalize_tool_result_text_fields(result: ToolResult) -> ToolResult:
    result.summary = _coerce_result_text(result.summary)
    result.evidence = _coerce_result_text(result.evidence)
    if result.error is not None:
        result.error = _coerce_result_text(result.error)
    return result


def _clip_log_text(value: Any) -> str:
    text = str(value or "").replace("\n", "\\n").strip()
    return text


def _tool_result_log_fields(result: ToolResult) -> tuple[str, str, str]:
    log_summary = ""
    if isinstance(result.metadata, dict):
        log_summary = str(result.metadata.get("log_summary") or "").strip()
    summary = _clip_log_text(log_summary or result.summary)
    error = _clip_log_text(result.error or "")
    state = ""
    if isinstance(result.state_delta, dict) and result.state_delta:
        try:
            state = _clip_log_text(json.dumps(result.state_delta, ensure_ascii=False, sort_keys=True))
        except Exception:
            state = _clip_log_text(result.state_delta)
    return summary, error, state


def _worker_log_fields(result: ToolResult) -> str:
    meta = result.metadata if isinstance(result.metadata, dict) else {}
    parts: list[str] = []
    path = str(meta.get("worker_path") or "").strip()
    if path:
        parts.append(f"path={path}")
    mode = str(meta.get("execution_mode") or "").strip()
    if mode:
        parts.append(f"mode={mode}")
    for key, label in (
        ("worker_limit", "limit"),
        ("worker_wait_ms", "wait_ms"),
        ("worker_inflight", "inflight"),
        ("worker_waiting", "queue"),
        ("worker_peak_inflight", "peak"),
        ("dispatch_ms", "dispatch_ms"),
    ):
        value = meta.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{label}={value}")
    monitor = meta.get("run_monitor")
    if isinstance(monitor, dict):
        kind = str(monitor.get("kind") or "").strip()
        if kind:
            parts.append(f"monitor={kind}")
    return " ".join(parts) or "-"


def _worker_limit_for_type(cfg: Config, worker_type: str) -> int:
    loop_cfg = getattr(cfg, "loop", None)
    attr_name = {
        "tool-chain-worker": "max_tool_chain_workers",
        "exec-worker": "max_exec_workers",
        "multimodal-worker": "max_multimodal_workers",
        "llm-worker": "max_llm_workers",
    }.get(worker_type, "max_tool_chain_workers")
    try:
        return max(1, int(getattr(loop_cfg, attr_name, 1) or 1))
    except (TypeError, ValueError):
        return 1


_TARGET_TASK_TOOLS = frozenset({
    "task.advance",
    "task.complete",
    "task.fail",
    "task.resume",
    "task.steer",
    "task.update",
    "task.wait",
})


def _coerce_task_id(value: Any) -> int:
    try:
        task_id = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return task_id if task_id > 0 else 0


def _planned_run_task_id(action: JudgmentOutput, active_task_id: int) -> int:
    tool_name = action.chosen_action_id or ""
    if tool_name in _TARGET_TASK_TOOLS:
        return _coerce_task_id((action.params or {}).get("task_id")) or active_task_id
    return active_task_id


def _resolved_run_task_id(result: ToolResult, active_task_id: int) -> int:
    tool_name = ""
    if isinstance(result.metadata, dict):
        tool_name = str(result.metadata.get("tool_name") or "")
    if tool_name in _TARGET_TASK_TOOLS and isinstance(result.metadata, dict):
        return _coerce_task_id(result.metadata.get("task_id")) or active_task_id
    return active_task_id


def _infer_run_profile(
    tool_name: str,
    params: dict[str, Any] | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> tuple[str, str]:
    p = params or {}
    if tool_name in {"evolution.evolve", "evolution.synthesize"}:
        return "evolve", "evolve-worker"
    if tool_name == "subagent.run":
        return "subagent", "subagent-worker"
    if p.get("monitor_fact_key") or p.get("status_fact_key"):
        _log.debug("[run-profile] tool=%s classified as llm-worker via fact monitor", tool_name)
        return "llm", "llm-worker"
    if tool_has_capability(registry, tool_name, "run_spawn"):
        return "exec", "exec-worker"
    if tool_has_capability(registry, tool_name, "multimodal"):
        return "multimodal", "multimodal-worker"
    return "tool_chain", "tool-chain-worker"


def _run_status_from_result(result: ToolResult) -> str:
    if (
        isinstance(result.state_delta, dict)
        and result.metadata.get("session_id")
        and result.state_delta.get("process") == "started"
        and result.state_delta.get("background")
    ):
        return "running"
    if result.error and not result.skipped:
        return "failed"
    if result.skipped:
        return "cancelled"
    return "succeeded"


def _run_progress_text(result: ToolResult) -> str:
    if isinstance(result.state_delta, dict):
        progress = str(result.state_delta.get("progress") or "").strip()
        if progress:
            return progress
    progress = str(result.metadata.get("progress") or "").strip()
    if progress:
        return progress
    log_summary = str(result.metadata.get("log_summary") or "").strip()
    if log_summary:
        return log_summary
    return (result.summary or "").strip()


async def _resolve_execution_active_task(ctx: ToolContext) -> Any:
    active_task = await ctx.get_active_task()
    if active_task is not None:
        return active_task
    task_store = getattr(ctx, "task_store", None)
    getter = getattr(task_store, "get_active", None)
    if getter is None:
        return None
    try:
        return await getter()
    except Exception:
        return None


def _meta_reflection_decision(target_kind: str, loop_level: str, text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("rollback", "回滚", "regressed", "regression")):
        return "rollback"
    if loop_level == "double" or target_kind in {"threshold", "routing", "task_split"}:
        return "apply"
    return "defer"


def _record_run_started(
    ctx: ToolContext,
    *,
    run_id: int,
    task_id: int,
    tool_name: str,
    run_type: str,
    worker_type: str,
    model_tier: str,
) -> None:
    if ctx.episodic is None:
        return
    ctx.episodic.record_event(
        "run_started",
        {
            "run_id": run_id,
            "task_id": task_id,
            "tool_name": tool_name,
            "run_type": run_type,
            "worker_type": worker_type,
            "model_tier": model_tier,
        },
    )


async def record_run_outcome_memory(
    episodic: Any | None,
    semantic: Any | None,
    *,
    memory_cfg: Any | None,
    run_id: int,
    task_id: int,
    tool_name: str,
    worker_type: str,
    status: str,
    progress: str,
    summary: str,
    error: str,
    evidence: str = "",
    owner: Any | None = None,
    task_store: Any | None = None,
    metabolic: Any | None = None,
) -> None:
    writer_owner = metabolic or owner or task_store
    if semantic is None and writer_owner is None and task_store is None:
        return
    is_failure = bool(error) or status == "failed"
    if episodic is not None:
        event_type = "run_failed" if is_failure or status == "failed" else "run_completed"
        episodic.record_event(
            event_type,
            {
                "run_id": run_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "worker_type": worker_type,
                "status": status,
                "summary": summary,
                "error": error,
            },
        )
    if semantic is None:
        return
    from store.semantic import MemoryNode

    tags = [status]
    if tool_name:
        tags.append(tool_name)
    if worker_type:
        tags.append(worker_type)
    if task_id:
        tags.append(f"task:{task_id}")
    if is_failure and "failed" not in tags:
        tags.append("failed")
    body_parts = [
        f"status={status}",
        f"tool={tool_name or 'unknown'}",
    ]
    if progress:
        body_parts.append(f"progress={_clip_run_outcome_memory_field(progress)}")
    if summary:
        body_parts.append(f"summary={_clip_run_outcome_memory_field(summary)}")
    if error:
        body_parts.append(f"error={_clip_run_outcome_memory_field(error)}")
    if evidence:
        body_parts.append(f"evidence={_clip_run_outcome_memory_field(evidence)}")
    activation, valence = run_result_memory_affect(
        memory_cfg,
        is_failure=is_failure,
    )
    # 优先走代谢提案写入语义记忆；失败时回退到原始 upsert，避免影响主流程可用性。
    try:
        await add_semantic_memory(
            writer_owner,
            task_store=task_store,
            semantic_memory=semantic,
            node_id=f"run-result-{run_id}",
            kind="run_result",
            title=f"[run#{run_id}] {tool_name or 'unknown'} {status}",
            body="\n".join(body_parts),
            activation=activation,
            valence=valence,
            tags=tags,
            source="execution/run_result",
            decision_basis=f"run_result_{run_id}",
        )
    except Exception:
        if semantic is not None:
            semantic.upsert(MemoryNode(
                id=f"run-result-{run_id}",
                kind="run_result",
                title=f"[run#{run_id}] {tool_name or 'unknown'} {status}",
                body="\n".join(body_parts),
                activation=activation,
                valence=valence,
                tags=tags,
            ))


async def record_meta_reflection_memory(
    episodic: Any | None,
    semantic: Any | None,
    meta: dict[str, str | int],
    *,
    owner: Any | None = None,
    task_store: Any | None = None,
    metabolic: Any | None = None,
) -> None:
    reflection_id = str(meta.get("reflection_id") or "")
    target_kind = str(meta.get("target_kind") or "")
    loop_level = str(meta.get("loop_level") or "")
    decision = str(meta.get("decision") or "defer")
    task_id = int(meta.get("task_id") or 0)
    run_id = int(meta.get("run_id") or 0)
    tool_name = str(meta.get("tool_name") or "")
    diagnosis = str(meta.get("diagnosis") or "")
    proposal = str(meta.get("proposal") or "")
    verification_plan = str(meta.get("verification_plan") or "")

    if episodic is not None and loop_level == "double":
        episodic.record_event(
            "double_loop_reflection",
            {
                "reflection_id": reflection_id,
                "run_id": run_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "target_kind": target_kind,
                "decision": decision,
            },
        )
    writer_owner = metabolic or owner or task_store
    if semantic is None:
        return
    from store.semantic import MemoryNode

    tags = ["meta_reflection", target_kind, loop_level, decision]
    if tool_name:
        tags.append(tool_name)
    if task_id:
        tags.append(f"task:{task_id}")
    # 优先走代谢提案；失败则回退到直接 upsert，保证既能审计也不影响反思链路。
    try:
        await add_semantic_memory(
            writer_owner,
            task_store=task_store,
            semantic_memory=semantic,
            node_id=f"meta-reflection-{reflection_id}",
            kind="meta_reflection",
            title=f"[{decision}] {target_kind or 'reflection'} run#{run_id}",
            body=(
                f"diagnosis={diagnosis}\n"
                f"proposal={proposal}\n"
                f"verification_plan={verification_plan}"
            ),
            activation=0.8 if decision != "defer" else 0.7,
            valence=0.42 if decision == "rollback" else 0.58,
            tags=tags,
            source="execution/meta_reflection",
            decision_basis=f"meta_reflection:{reflection_id}",
        )
    except Exception:
        semantic.upsert(MemoryNode(
            id=f"meta-reflection-{reflection_id}",
            kind="meta_reflection",
            title=f"[{decision}] {target_kind or 'reflection'} run#{run_id}",
            body=(
                f"diagnosis={diagnosis}\n"
                f"proposal={proposal}\n"
                f"verification_plan={verification_plan}"
            ),
            activation=0.8 if decision != "defer" else 0.7,
            valence=0.42 if decision == "rollback" else 0.58,
            tags=tags,
        ))
    if decision in {"apply", "rollback"}:
        rule_target = target_kind or "rule"
        rule_tool = tool_name or "unknown-tool"
        # apply/rollback 形成规则修订时，同步落地到语义层；异常回退避免丢事件。
        try:
            await add_semantic_memory(
                writer_owner,
                task_store=task_store,
                semantic_memory=semantic,
                node_id=f"rule-revision-{reflection_id}",
                kind="rule_revision",
                title=f"[{decision}] {rule_target} via {rule_tool} run#{run_id}",
                body=(
                    f"target_kind={target_kind}\n"
                    f"tool_name={tool_name}\n"
                    f"proposal={proposal}\n"
                    f"verification_plan={verification_plan}"
                ),
                activation=0.83,
                valence=0.46 if decision == "rollback" else 0.62,
                tags=[target_kind, decision, tool_name or ""] if tool_name else [target_kind, decision],
                source="execution/rule_revision",
                decision_basis=f"rule_revision:{reflection_id}",
            )
        except Exception:
            semantic.upsert(MemoryNode(
                id=f"rule-revision-{reflection_id}",
                kind="rule_revision",
                title=f"[{decision}] {rule_target} via {rule_tool} run#{run_id}",
                body=(
                    f"target_kind={target_kind}\n"
                    f"tool_name={tool_name}\n"
                    f"proposal={proposal}\n"
                    f"verification_plan={verification_plan}"
                ),
                activation=0.83,
                valence=0.46 if decision == "rollback" else 0.62,
                tags=[target_kind, decision, tool_name or ""] if tool_name else [target_kind, decision],
            ))


_LOW_VALUE_SUCCESS_TOOLS = frozenset({
    "file.read", "file.list", "memory.search", "memory.get_fact", "memory.list_facts",
    "probe.list", "probe.run", "schedule.list", "task.list", "config.get", "config.list_keys",
    "shell.capabilities", "skill.list", "skill.search", "skill.activate",
    "browser.snapshot", "web.fetch", "web.search", "image.analyze",
})

def _should_record_successful_run(tool_name: str) -> bool:
    """Check if a successful run should be recorded to semantic memory."""
    return tool_name not in _LOW_VALUE_SUCCESS_TOOLS

def _should_record_run_outcome(status: str) -> bool:
    return status in {"succeeded", "failed", "cancelled"}


def build_meta_reflection(
    *,
    run_id: int,
    task_id: int,
    tool_name: str,
    result: ToolResult,
) -> dict[str, str | int] | None:
    if not (result.error or result.skipped):
        return None
    text = "\n".join(x for x in [tool_name, result.error or "", result.summary, result.evidence] if x).lower()
    target_kind = "tool"
    trigger = "failure_pattern"
    loop_level = "single"
    diagnosis = f"动作 {tool_name or 'unknown'} 在 run#{run_id} 结束为 {_run_status_from_result(result)}，需要复盘失败来源。"
    proposal = "优先检查工具实现、输入参数或外部资源，然后重试同一动作。"
    verification_plan = "在相同 task 上用同一输入重跑一次，确认错误消失或 summary 改善。"

    if "knownstablefailure" in text:
        target_kind = "threshold"
        diagnosis = f"动作 {tool_name or 'unknown'} 被稳定失败降噪机制拦截，说明当前静默阈值或外部状态需要复查。"
        proposal = "确认外部状态是否恢复；若频繁误杀，则调整 durable failure 阈值或静默策略。"
        verification_plan = "等待静默窗口结束后重跑，并比较是否仍被直接跳过。"
    elif "emptypath" in text or "path 不能为空" in text:
        target_kind = "task_split"
        loop_level = "double"
        diagnosis = f"动作 {tool_name or 'unknown'} 缺少必要资源定位，问题更像任务拆分不完整，而不只是工具报错。"
        proposal = "在创建读取/写入类 run 之前，先增加资源发现或路径确认步骤，再执行目标动作。"
        verification_plan = "先补一条定位资源的子步骤，再重跑原动作，确认不再出现空路径错误。"
    elif "toolnotfound" in text or "工具不存在" in text:
        target_kind = "routing"
        loop_level = "double"
        diagnosis = f"动作 {tool_name or 'unknown'} 未注册，说明判断层的动作选择或工具清单存在漂移。"
        proposal = "校正 action 选择规则或工具清单注入，避免继续选择不存在的动作。"
        verification_plan = "重新做一次 judgment，确认 chosen_action_id 落在已注册工具集合内。"
    decision = _meta_reflection_decision(target_kind, loop_level, text)

    return {
        "reflection_id": f"mr-{uuid.uuid4().hex[:12]}",
        "target_kind": target_kind,
        "trigger": trigger,
        "loop_level": loop_level,
        "diagnosis": diagnosis,
        "proposal": proposal,
        "verification_plan": verification_plan,
        "decision": decision,
        "task_id": task_id,
        "run_id": run_id,
        "tool_name": tool_name,
    }


async def finalize_run(
    *,
    cfg: Config,
    run_id: int | None,
    result: ToolResult,
    ctx: ToolContext,
    active_task_id: int | None = None,
) -> None:
    if run_id is None or ctx.task_store is None:
        return
    result.metadata.setdefault("run_id", run_id)
    if isinstance(result.state_delta, dict):
        result.state_delta.setdefault("run_id", run_id)
    resolved_task_id = _resolved_run_task_id(result, active_task_id or 0)
    status = _run_status_from_result(result)
    progress = _run_progress_text(result)
    await update_run(
        ctx,
        run_id,
        task_id=resolved_task_id,
        status=status,
        output_json=result.to_dict(),
        log_text=result.summary,
        error_text=result.error or "",
        session_id=str(result.metadata.get("session_id") or ""),
        progress=progress,
        source="execution/helpers/finalize_run",
        decision_basis=f"run_id={run_id} status={status}",
    )
    _summary_log, _error_log, _state_log = _tool_result_log_fields(result)
    _worker_log = _worker_log_fields(result)
    scope = execution_scope_fields(
        run_id=run_id,
        task_id=resolved_task_id,
        tool=str(result.metadata.get("tool_name") or ""),
        worker=str(result.metadata.get("worker_type") or ""),
        status=status,
    )
    _log.info(
        "[run-finalize] %s worker_meta=%s progress=%s error=%s state=%s",
        scope,
        _worker_log,
        _clip_log_text(progress or "") or "-",
        _error_log or "-",
        _state_log or "-",
    )
    if _should_record_run_outcome(status):
        tool_name_for_filter = str(result.metadata.get("tool_name") or "")
        if status == "succeeded" and not _should_record_successful_run(tool_name_for_filter):
            _log.debug("[run-finalize] Skipping semantic memory for low-value successful run: %s", tool_name_for_filter)
        else:
            await record_run_outcome_memory(
                ctx.episodic,
                ctx.semantic,
                memory_cfg=getattr(ctx.config, "memory", None),
                run_id=run_id,
                task_id=resolved_task_id,
                tool_name=tool_name_for_filter,
                worker_type=str(result.metadata.get("worker_type") or ""),
                status=status,
                progress=progress,
                summary=result.summary,
                error=result.error or "",
                evidence=result.evidence,
                task_store=ctx.task_store,
            )
    if resolved_task_id:
        await update_task_result(
            ctx,
            resolved_task_id,
            build_task_run_result_patch(
                run_id=run_id,
                status=status,
                worker_type=str(result.metadata.get("worker_type") or ""),
                tool_name=str(result.metadata.get("tool_name") or ""),
                session_id=str(result.metadata.get("session_id") or ""),
                summary=result.summary,
                error=result.error,
            ),
            source="execution/helpers/finalize_run",
        )
    meta = build_meta_reflection(
        run_id=run_id,
        task_id=resolved_task_id,
        tool_name=str(result.metadata.get("tool_name") or ""),
        result=result,
    )
    if meta:
        await ctx.task_store.add_meta_reflection(
            reflection_id=str(meta["reflection_id"]),
            target_kind=str(meta["target_kind"]),
            trigger=str(meta["trigger"]),
            loop_level=str(meta["loop_level"]),
            diagnosis=str(meta["diagnosis"]),
            proposal=str(meta["proposal"]),
            verification_plan=str(meta["verification_plan"]),
            decision=str(meta["decision"]),
            task_id=int(meta["task_id"]),
            run_id=int(meta["run_id"]),
            tool_name=str(meta["tool_name"]),
        )
        await record_meta_reflection_memory(
            ctx.episodic,
            ctx.semantic,
            meta,
            task_store=ctx.task_store,
        )
