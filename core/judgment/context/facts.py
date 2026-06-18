"""Judgment context section loaders for fact snapshots."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from store.task import Task, TaskStore


_FACT_CONTEXT_EXCLUDE_PREFIXES = (
    "control:",
    "durable_failure:",
    "evolution:",
    "pref:",
    "run:",
    "soul:",
)


def _should_include_fact_key(
    key: str,
    *,
    seen: set[str],
    task_prefix: str,
    exclude_prefixes: tuple[str, ...],
) -> bool:
    if key in seen:
        return False
    if exclude_prefixes and key.startswith(exclude_prefixes):
        return False
    if key.startswith("task:") and task_prefix and not key.startswith(task_prefix):
        return False
    if key.startswith("task:") and not task_prefix:
        return False
    return True


async def _load_context_facts_snapshot(
    task_store: TaskStore,
    task: Task | None,
    *,
    exclude_prefixes: list[str] | tuple[str, ...] | None = None,
    task_limit: int = 6,
    global_limit: int = 4,
    priority_prefixes: list[str] | tuple[str, ...] | None = None,
    priority_limit: int = 2,
    recent_scan_multiplier: int = 3,
    recent_scan_min: int = 12,
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    selected: list[tuple[str, str]] = []
    task_prefix = f"task:{task.id}:" if task else ""
    exclude_prefixes_tuple = tuple(exclude_prefixes) if exclude_prefixes is not None else _FACT_CONTEXT_EXCLUDE_PREFIXES
    priority_prefixes_tuple = tuple(priority_prefixes or ())

    async def _add_facts(items: list[tuple[str, str]], limit: int) -> int:
        added = 0
        for key, value in items:
            if not _should_include_fact_key(
                key,
                seen=seen,
                task_prefix=task_prefix,
                exclude_prefixes=exclude_prefixes_tuple,
            ):
                continue
            seen.add(key)
            selected.append((key, value))
            added += 1
            if limit > 0 and added >= limit:
                return added
        return added

    if task_prefix:
        task_facts = await task_store.list_facts(prefix=task_prefix, limit=task_limit)
        await _add_facts(task_facts, task_limit)

    if priority_prefixes_tuple and priority_limit > 0:
        remaining = priority_limit
        for prefix in priority_prefixes_tuple:
            if remaining <= 0:
                break
            priority_facts = await task_store.list_facts(prefix=prefix, limit=remaining)
            added = await _add_facts(priority_facts, remaining)
            remaining -= added

    if global_limit > 0:
        recent_scan_limit = max(global_limit * recent_scan_multiplier, recent_scan_min)
        recent_facts = await task_store.list_facts(limit=recent_scan_limit)
        before = len(selected)
        await _add_facts(recent_facts, global_limit)
        if len(selected) == before and not selected:
            return []

    return selected


async def _load_durable_failure_snapshot(task_store: TaskStore) -> dict[str, Any]:
    from core.execution import _load_durable_failure_policy

    policy = await _load_durable_failure_policy(task_store)
    muted_actions: list[dict[str, Any]] = []
    now = time.time()
    for _, raw in await task_store.list_facts(prefix="durable_failure:", limit=12):
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        muted_until = float(payload.get("muted_until") or 0)
        if muted_until <= now:
            continue
        muted_actions.append({
            "tool": str(payload.get("tool") or ""),
            "key": str(payload.get("key") or "").strip(),
            "reason": str(payload.get("reason") or "stable_failure"),
            "count": int(payload.get("count") or 0),
            "remaining_sec": max(0, int(muted_until - now)),
        })
    muted_actions.sort(key=lambda item: item["remaining_sec"])
    return {
        "threshold": int(policy.get("threshold") or 0),
        "ttl_sec": int(policy.get("ttl_sec") or 0),
        "muted_actions": muted_actions,
    }


__all__ = [
    "_FACT_CONTEXT_EXCLUDE_PREFIXES",
    "_load_context_facts_snapshot",
    "_load_durable_failure_snapshot",
]
