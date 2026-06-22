"""core/metabolic/engine.py — 代谢引擎（正式状态写入唯一出口）。

公理 A5：正式状态写入必须经过代谢器官。

当前职责：
  1. 接收外围器官提交的 StateProposal
  2. 先经免疫器官检查
  3. 将允许的提案落地到 TaskStore
  4. 将通过、拒绝、未知 op、落地失败全部追加到生命史账本
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from core.immune.policy import check_tool_blocked
from core.metabolic.lifecycle_utils import decision_basis_from_parts
from core.metabolic.state_writer import apply_state_write

if TYPE_CHECKING:
    from core.metabolic.proposal import StateProposal
    from tools.view_protocols import TaskStoreViewProtocol

_log = logging.getLogger("lingzhou.metabolic")

_FACT_IMMUNE_OPS = {"set_fact", "delete_fact"}
_LEDGER_VALUE_MAX_CHARS = 16000


class MetabolicEngine:
    """代谢引擎：全系统唯一正式状态写入出口。"""

    def __init__(self, task_store: TaskStoreViewProtocol, semantic_memory: Any | None = None) -> None:
        self._task_store = task_store
        self._semantic_memory = semantic_memory

    async def submit(self, proposal: StateProposal) -> Any:
        """提交候选状态写入。

        流程：
          1. 免疫器官检查（fact key 映射为伪工具名 "fact:<key>"）
          2. 通过 → 按 proposal.op 落地写入
          3. 无论通过/拒绝/失败，均追加生命史账本（accepted 标记区分）
        """
        # fact 写入/删除以 "fact:<key>" 形式过黑名单；soul_change 等高风险 op
        # 日后可在 check_tool_blocked 中新增规则，此处自动生效。
        block_reason = check_tool_blocked(_pseudo_tool_name(proposal))

        accepted = block_reason is None
        result = None
        ledger_key = proposal.key
        ledger_reason = ""
        write_error: Exception | None = None

        if not accepted:
            ledger_reason = str(block_reason or "immune_blocked")
            _log_blocked_write(proposal, block_reason)
        else:
            try:
                applied = await apply_state_write(
                    self._task_store,
                    proposal,
                    accepted=accepted,
                    semantic_memory=self._semantic_memory,
                )
                result = applied.result
                ledger_key = applied.ledger_key
                accepted = applied.accepted
                ledger_reason = applied.reason
            except Exception as exc:
                accepted = False
                ledger_reason = _write_error_reason(exc)
                write_error = exc
                _log.warning(
                    "[metabolic] 落地失败 key=%r op=%r source=%r error=%s",
                    proposal.key,
                    proposal.op,
                    proposal.source,
                    exc,
                )

        await _append_lifecycle_ledger(
            self._task_store,
            proposal,
            ledger_key=ledger_key,
            accepted=accepted,
            ledger_reason=ledger_reason,
        )
        if write_error is not None:
            raise write_error
        return result


def _pseudo_tool_name(proposal: StateProposal) -> str:
    return f"fact:{proposal.key}" if proposal.op in _FACT_IMMUNE_OPS else proposal.op


def _log_blocked_write(proposal: StateProposal, block_reason: Any) -> None:
    _log.warning(
        "[metabolic] 免疫器官拒绝写入 key=%r op=%r source=%r reason=%s",
        proposal.key,
        proposal.op,
        proposal.source,
        block_reason,
    )


async def _append_lifecycle_ledger(
    task_store: TaskStoreViewProtocol,
    proposal: StateProposal,
    *,
    ledger_key: str,
    accepted: bool,
    ledger_reason: str,
) -> None:
    try:
        await task_store.ledger_append(
            op=proposal.op,
            key=ledger_key,
            value=_ledger_value(proposal.value),
            scope=proposal.scope,
            source=proposal.source,
            accepted=accepted,
            run_id=proposal.run_id,
            reason=ledger_reason,
            proposal_hash=_proposal_hash(proposal),
            decision_basis=decision_basis_from_parts(
                proposal.extras.get("decision_basis") or proposal.extras.get("basis") or "",
                limit=1000,
            ),
        )
    except Exception as exc:
        _log.warning("[metabolic] 生命史账本写入失败（不影响主流程）: %s", exc)


def _ledger_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    return _clip_ledger_text(text)


def _clip_ledger_text(text: str, *, limit: int = _LEDGER_VALUE_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    marker = f"\n...[life_ledger value truncated chars={len(text)} sha256={digest}]...\n"
    budget = max(0, limit - len(marker))
    head = max(200, budget // 2)
    tail = max(0, budget - head)
    return text[:head] + marker + (text[-tail:] if tail else "")


def _proposal_hash(proposal: StateProposal) -> str:
    payload = {
        "op": proposal.op,
        "key": proposal.key,
        "value": _proposal_hash_value(proposal.value),
        "scope": proposal.scope,
        "source": proposal.source,
        "run_id": proposal.run_id,
        "extras": proposal.extras,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _proposal_hash_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _LEDGER_VALUE_MAX_CHARS:
        return {
            "storage": "sha256",
            "chars": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): _proposal_hash_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_proposal_hash_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_proposal_hash_value(item) for item in value)
    return value


def _write_error_reason(exc: Exception) -> str:
    return decision_basis_from_parts(
        "write_error",
        exc.__class__.__name__,
        exc,
        limit=500,
    )
