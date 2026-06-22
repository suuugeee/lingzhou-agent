"""状态落地器：把已通过免疫检查的 StateProposal 写入 TaskStore。"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.metabolic.proposal import StateProposal
    from tools.view_protocols import TaskStoreViewProtocol

_log = logging.getLogger("lingzhou.metabolic")

_DEFAULT_RUN_TYPE = "tool_chain"
_DEFAULT_WORKER_TYPE = "tool-chain-worker"
_PERSISTENT_TEXT_MAX_CHARS = 12000
_PERSISTENT_COLLECTION_MAX_ITEMS = 80
_SEMANTIC_DIRECT_BLOCKED_KINDS = frozenset({
    "run_result",
    "task_progress",
    "working_trace",
    "execute_result",
    "run_monitor",
})
_SEMANTIC_DIRECT_SOURCES = frozenset({
    "tools/memory.add_semantic",
})
_RAW_OPERATIONAL_TEXT_MARKERS = (
    "stdout",
    "stderr",
    "traceback",
    "process exited with code",
    "exit code",
    "wall time",
    "run_id",
    "tool=",
    "status=",
    "[execute_result]",
    "[run 监控]",
    "command:",
)
_FACT_WRITE_OPS = {"set_fact", "delete_fact"}
_TASK_WRITE_OPS = {
    "create_task",
    "update_task_status",
    "mark_task_waiting",
    "resume_task",
    "update_task_data",
    "update_task_result",
    "amend_task",
    "add_run",
    "update_run",
}


@dataclass(slots=True)
class StateWriteResult:
    result: Any = None
    ledger_key: str = ""
    accepted: bool = True
    reason: str = ""


async def apply_state_write(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    accepted: bool,
    semantic_memory: Any | None = None,
) -> StateWriteResult:
    """落地一次已获准的状态写入；未知 op 返回 accepted=False。"""
    if proposal.op in _FACT_WRITE_OPS:
        return await _apply_fact_write(task_store, proposal, accepted=accepted)
    if proposal.op == "soul_change":
        return await _apply_soul_change(task_store, proposal, accepted=accepted)
    if proposal.op in _TASK_WRITE_OPS:
        return await _apply_task_write(task_store, proposal, accepted=accepted)
    if proposal.op == "add_semantic_memory":
        return await _apply_semantic_write(
            semantic_memory,
            proposal,
            accepted=accepted,
        )

    _log.warning(
        "[metabolic] 未知 op=%r，跳过（key=%r source=%r）",
        proposal.op,
        proposal.key,
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=False, reason="unknown_op")


async def _apply_fact_write(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    accepted: bool,
) -> StateWriteResult:
    if proposal.op == "set_fact":
        await task_store.set_fact(
            proposal.key,
            proposal.value,
            scope=proposal.scope,
        )
        _log.debug(
            "[metabolic] set_fact key=%r scope=%r source=%r",
            proposal.key,
            proposal.scope,
            proposal.source,
        )
        return StateWriteResult(ledger_key=proposal.key, accepted=accepted)

    await task_store.delete_fact(proposal.key)
    _log.debug(
        "[metabolic] delete_fact key=%r source=%r",
        proposal.key,
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_soul_change(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    accepted: bool,
) -> StateWriteResult:
    """人格/灵魂层 fact 落地：key 必须是 soul 前缀。"""
    if not str(proposal.key).startswith("soul:"):
        return StateWriteResult(
            ledger_key=proposal.key,
            accepted=False,
            reason="soul_change key must start with 'soul:'",
        )
    await task_store.set_fact(
        proposal.key,
        proposal.value,
        scope=proposal.scope,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_task_write(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    accepted: bool,
) -> StateWriteResult:
    data = _proposal_data(proposal)
    handler = {
        "create_task": _apply_create_task,
        "update_task_status": _apply_update_task_status,
        "mark_task_waiting": _apply_mark_task_waiting,
        "resume_task": _apply_resume_task,
        "update_task_data": _apply_update_task_data,
        "update_task_result": _apply_update_task_result,
        "add_run": _apply_add_run,
        "update_run": _apply_update_run,
    }.get(proposal.op, _apply_amend_task)
    return await handler(task_store, proposal, data=data, accepted=accepted)


async def _apply_create_task(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    task_id = await task_store.add_task(**data)
    _log.debug(
        "[metabolic] create_task id=%r source=%r",
        task_id,
        proposal.source,
    )
    return StateWriteResult(result=task_id, ledger_key=f"task:{task_id}", accepted=accepted)


async def _apply_update_task_status(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    await task_store.update_status(
        _proposal_task_id(proposal),
        _str_field(data, "status"),
        data.get("next_step"),
        current_step=data.get("current_step"),
        model_tier=data.get("model_tier"),
        result_json=data.get("result_json"),
    )
    _log.debug(
        "[metabolic] update_task_status task_id=%r status=%r source=%r",
        proposal.key,
        data.get("status"),
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_mark_task_waiting(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    await task_store.mark_waiting(
        _proposal_task_id(proposal),
        wait_kind=_str_field(data, "wait_kind"),
        wait_key=_str_field(data, "wait_key"),
        wait_json=data.get("wait_json"),
        current_step=data.get("current_step"),
        next_step=data.get("next_step"),
        result_json=data.get("result_json"),
    )
    _log.debug(
        "[metabolic] mark_task_waiting task_id=%r wait_kind=%r source=%r",
        proposal.key,
        data.get("wait_kind"),
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_resume_task(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    await task_store.resume_task(
        _proposal_task_id(proposal),
        status=_str_field(data, "status", default="resumed"),
        current_step=data.get("current_step"),
        next_step=data.get("next_step"),
        result_json=data.get("result_json"),
    )
    _log.debug(
        "[metabolic] resume_task task_id=%r status=%r source=%r",
        proposal.key,
        data.get("status") or "resumed",
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_update_task_data(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    await task_store.update_task_data(_proposal_task_id(proposal), data)
    _log.debug(
        "[metabolic] update_task_data task_id=%r source=%r",
        proposal.key,
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_update_task_result(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    await task_store.update_task_result(
        _proposal_task_id(proposal),
        _compact_persistent_mapping(_task_result_value(proposal)),
    )
    _log.debug(
        "[metabolic] update_task_result task_id=%r source=%r",
        proposal.key,
        proposal.source,
    )
    return StateWriteResult(ledger_key=proposal.key, accepted=accepted)


async def _apply_add_run(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    run_id = await task_store.add_run(
        task_id=_int_field(data, "task_id"),
        run_type=_str_field(data, "run_type", default=_DEFAULT_RUN_TYPE),
        worker_type=_str_field(data, "worker_type", default=_DEFAULT_WORKER_TYPE),
        status=_str_field(data, "status", default="running"),
        input_json=_compact_persistent_mapping(_dict_field(data, "input_json", default={}) or {}),
        output_json=_compact_persistent_mapping(_dict_field(data, "output_json", default={}) or {}),
        log_text=_clip_persistent_text(_str_field(data, "log_text")),
        error_text=_clip_persistent_text(_str_field(data, "error_text")),
        tool_name=_str_field(data, "tool_name"),
        session_id=_str_field(data, "session_id"),
        model_tier=_str_field(data, "model_tier"),
        progress=_clip_persistent_text(_str_field(data, "progress")),
        extras=_compact_persistent_mapping(_dict_field(data, "extras", default={}) or {}),
    )
    _log.debug(
        "[metabolic] add_run key=%r run_id=%r source=%r",
        proposal.key,
        run_id,
        proposal.source,
    )
    return StateWriteResult(result=run_id, ledger_key=f"run:{run_id}", accepted=accepted)


async def _apply_update_run(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    await task_store.update_run(
        _proposal_task_id(proposal),
        task_id=data.get("task_id") if data.get("task_id") is not None else None,
        status=_optional_str_field(data, "status"),
        output_json=_compact_optional_mapping(_dict_field(data, "output_json")),
        log_text=_clip_optional_persistent_text(_optional_str_field(data, "log_text")),
        error_text=_clip_optional_persistent_text(_optional_str_field(data, "error_text")),
        session_id=_optional_str_field(data, "session_id"),
        model_tier=_optional_str_field(data, "model_tier"),
        progress=_clip_optional_persistent_text(_optional_str_field(data, "progress")),
        extras=_compact_optional_mapping(_dict_field(data, "extras")),
    )
    _log.debug(
        "[metabolic] update_run run_id=%r status=%r source=%r",
        proposal.key,
        data.get("status"),
        proposal.source,
    )
    return StateWriteResult(ledger_key=f"run:{proposal.key}", accepted=accepted)


async def _apply_amend_task(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    data: dict[str, Any],
    accepted: bool,
) -> StateWriteResult:
    result = await task_store.amend_task(
        _proposal_task_id(proposal),
        title=data.get("title"),
        goal=data.get("goal"),
        priority=data.get("priority"),
        amendment_reason=_str_field(data, "amendment_reason"),
    )
    _log.debug(
        "[metabolic] amend_task task_id=%r accepted=%r source=%r",
        proposal.key,
        result,
        proposal.source,
    )
    return StateWriteResult(result=result, ledger_key=proposal.key, accepted=accepted)


async def _apply_semantic_write(
    semantic_memory: Any | None,
    proposal: StateProposal,
    *,
    accepted: bool,
) -> StateWriteResult:
    if semantic_memory is None or not callable(getattr(semantic_memory, "upsert", None)):
        return StateWriteResult(
            ledger_key=proposal.key,
            accepted=False,
            reason="semantic_memory_unavailable",
        )
    data = _proposal_data(proposal)
    kind = str(data.get("kind") or "observation")
    source = str(data.get("source") or proposal.source or "")
    title = str(data.get("title") or "")
    body = str(data.get("body") or "")
    if _semantic_write_is_operational_noise(kind, source):
        return StateWriteResult(
            ledger_key=proposal.key,
            accepted=False,
            reason=f"semantic_operational_noise:{kind}",
        )
    if _semantic_write_has_raw_operational_body(title, body, source):
        return StateWriteResult(
            ledger_key=proposal.key,
            accepted=False,
            reason="semantic_operational_body",
        )
    if _semantic_write_is_low_value(kind, title, body, source):
        return StateWriteResult(
            ledger_key=proposal.key,
            accepted=False,
            reason="semantic_low_value_process_note",
        )
    from store.semantic import MemoryNode

    node = MemoryNode(
        id=str(data.get("id") or proposal.key),
        kind=kind,
        title=title,
        body=_clip_persistent_text(body),
        activation=float(data.get("activation", 0.5)),
        valence=float(data.get("valence", 0.5)),
        importance=float(data.get("importance", 0.0)),
        tags=[str(tag) for tag in data.get("tags", [])] if isinstance(data.get("tags"), list) else [],
        source=source,
        created_at=_created_at(data),
    )
    semantic_memory.upsert(node)
    _log.debug(
        "[metabolic] add_semantic_memory node_id=%r source=%r",
        node.id,
        proposal.source,
    )
    return StateWriteResult(result=node.id, ledger_key=f"semantic:{node.id}", accepted=accepted)


def _semantic_write_is_operational_noise(kind: str, source: str) -> bool:
    if kind not in _SEMANTIC_DIRECT_BLOCKED_KINDS:
        return False
    if source in _SEMANTIC_DIRECT_SOURCES or source.startswith("subagent:"):
        return True
    if source == "loop/consolidation" and kind in {"working_trace", "execute_result", "run_monitor"}:
        return True
    return False


def _semantic_write_has_raw_operational_body(title: str, body: str, source: str) -> bool:
    if source not in _SEMANTIC_DIRECT_SOURCES and not source.startswith("subagent:"):
        return False
    return _looks_like_raw_operational_text(title, body)


def _semantic_write_is_low_value(kind: str, title: str, body: str, source: str) -> bool:
    from memory.consolidation import is_low_value_semantic_text

    return is_low_value_semantic_text(kind, title, body)


def _looks_like_raw_operational_text(title: str, body: str) -> bool:
    text = f"{title}\n{body}".lower()
    body_text = str(body or "")
    marker_hits = sum(1 for marker in _RAW_OPERATIONAL_TEXT_MARKERS if marker in text)
    if marker_hits <= 0:
        return False

    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    line_count = len(lines)
    long_lines = sum(1 for line in lines if len(line) > 240)
    jsonish_lines = sum(1 for line in lines if line.startswith(("{", "[")) and line.endswith(("}", "]")))
    if marker_hits >= 2 and len(body_text) > 200:
        return True
    if len(body_text) > 1200 and (line_count >= 8 or long_lines >= 3):
        return True
    return line_count >= 40 and jsonish_lines >= 8


def _proposal_data(proposal: StateProposal) -> dict[str, Any]:
    return proposal.value if isinstance(proposal.value, dict) else {}


def _proposal_task_id(proposal: StateProposal) -> int:
    return int(proposal.key)


def _task_result_value(proposal: StateProposal) -> dict[str, Any]:
    return proposal.value if isinstance(proposal.value, dict) else {"value": proposal.value}


def _clip_persistent_text(value: Any, *, limit: int = _PERSISTENT_TEXT_MAX_CHARS) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    marker = f"\n...[persistent storage truncated chars={len(text)} sha256={digest}]...\n"
    budget = max(0, limit - len(marker))
    head = max(200, budget // 2)
    tail = max(0, budget - head)
    return text[:head] + marker + (text[-tail:] if tail else "")


def _clip_optional_persistent_text(value: str | None) -> str | None:
    return _clip_persistent_text(value) if value is not None else None


def _compact_persistent_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _clip_persistent_text(value)
    if isinstance(value, dict):
        if depth >= 5:
            return _clip_persistent_text(_json_preview(value))
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_PERSISTENT_COLLECTION_MAX_ITEMS]:
            compacted[str(key)] = _compact_persistent_value(item, depth=depth + 1)
        omitted = len(items) - len(compacted)
        if omitted > 0:
            compacted["_persistent_omitted_items"] = omitted
        return compacted
    if isinstance(value, (list, tuple)):
        if depth >= 5:
            return _clip_persistent_text(_json_preview(value))
        items = list(value)
        if len(items) <= _PERSISTENT_COLLECTION_MAX_ITEMS:
            return [
                _compact_persistent_value(item, depth=depth + 1)
                for item in items
            ]
        retained_items = max(2, _PERSISTENT_COLLECTION_MAX_ITEMS - 1)
        head_count = max(1, retained_items // 2)
        tail_count = max(1, retained_items - head_count)
        omitted = len(items) - head_count - tail_count
        return [
            *[
                _compact_persistent_value(item, depth=depth + 1)
                for item in items[:head_count]
            ],
            {"_persistent_omitted_items": omitted},
            *[
                _compact_persistent_value(item, depth=depth + 1)
                for item in items[-tail_count:]
            ],
        ]
    return value


def _compact_persistent_mapping(value: dict[str, Any]) -> dict[str, Any]:
    compacted = _compact_persistent_value(value)
    return compacted if isinstance(compacted, dict) else {}


def _compact_optional_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return _compact_persistent_mapping(value) if value is not None else None


def _json_preview(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _dict_field(
    data: dict[str, Any],
    key: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    value = data.get(key)
    if isinstance(value, dict):
        return value
    return default


def _int_field(data: dict[str, Any], key: str, *, default: int = 0) -> int:
    return int(data.get(key) or default)


def _str_field(data: dict[str, Any], key: str, *, default: str = "") -> str:
    return str(data.get(key) or default)


def _optional_str_field(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    return str(value) if value is not None else None


def _created_at(data: dict[str, Any]) -> str:
    return str(data.get("created_at") or "").strip() or datetime.now(UTC).isoformat()
