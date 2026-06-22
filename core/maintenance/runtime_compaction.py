"""Runtime DB compaction for oversized operational payloads."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

_TEXT_MAX_CHARS = 12000
_LEDGER_MAX_CHARS = 16000
_COLLECTION_MAX_ITEMS = 80


def compact_runtime_db(
    db_path: Path | str,
    *,
    apply: bool = False,
    vacuum: bool = False,
    text_max_chars: int = _TEXT_MAX_CHARS,
    ledger_max_chars: int = _LEDGER_MAX_CHARS,
) -> dict[str, Any]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return {
            "db_path": str(path),
            "dry_run": not apply,
            "error": "runtime_db_not_found",
            "scanned_rows": 0,
            "changed_rows": 0,
            "original_bytes": 0,
            "compacted_bytes": 0,
            "saved_bytes": 0,
            "tables": {},
            "vacuumed": False,
        }

    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        before_size = path.stat().st_size
        tables: dict[str, dict[str, int]] = {}
        totals = {"scanned_rows": 0, "changed_rows": 0, "original_bytes": 0, "compacted_bytes": 0}

        _compact_json_table(
            conn,
            table="runs",
            id_column="id",
            data_column="data",
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )
        _compact_json_table(
            conn,
            table="tasks",
            id_column="id",
            data_column="data",
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )
        _compact_json_table(
            conn,
            table="failures",
            id_column="id",
            data_column="data",
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )
        _compact_fact_table(
            conn,
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )
        _compact_chat_messages(
            conn,
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )
        _compact_meta_reflections(
            conn,
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )
        _compact_text_table(
            conn,
            table="life_ledger",
            id_column="id",
            data_column="value",
            tables=tables,
            totals=totals,
            apply=apply,
            text_max_chars=ledger_max_chars,
            marker_label="life_ledger value",
        )

        vacuumed = False
        if apply:
            conn.commit()
            if vacuum:
                conn.execute("VACUUM")
                vacuumed = True
        else:
            conn.rollback()
        after_size = path.stat().st_size if apply and vacuumed else before_size
        return {
            "db_path": str(path),
            "dry_run": not apply,
            "scanned_rows": totals["scanned_rows"],
            "changed_rows": totals["changed_rows"],
            "original_bytes": totals["original_bytes"],
            "compacted_bytes": totals["compacted_bytes"],
            "saved_bytes": max(0, totals["original_bytes"] - totals["compacted_bytes"]),
            "file_bytes_before": before_size,
            "file_bytes_after": after_size,
            "tables": tables,
            "vacuumed": vacuumed,
        }
    finally:
        conn.close()


def _compact_json_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    data_column: str,
    tables: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    if not _table_exists(conn, table):
        return
    stats = _stats_bucket(tables, table)
    rows = conn.execute(f"SELECT {id_column}, {data_column} FROM {table}").fetchall()
    for row in rows:
        raw = str(row[data_column] or "")
        compacted = _compact_json_text(raw, text_max_chars=text_max_chars)
        _record_compaction(stats, totals, raw, compacted)
        if apply and compacted != raw:
            conn.execute(
                f"UPDATE {table} SET {data_column}=? WHERE {id_column}=?",
                (compacted, row[id_column]),
            )


def _compact_fact_table(
    conn: sqlite3.Connection,
    *,
    tables: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    if not _table_exists(conn, "facts"):
        return
    stats = _stats_bucket(tables, "facts")
    rows = conn.execute("SELECT key, value FROM facts").fetchall()
    for row in rows:
        key = str(row["key"] or "")
        raw = str(row["value"] or "")
        compacted = _compact_fact_value(key, raw, text_max_chars=text_max_chars)
        _record_compaction(stats, totals, raw, compacted)
        if apply and compacted != raw:
            conn.execute("UPDATE facts SET value=? WHERE key=?", (compacted, key))


def _compact_text_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    data_column: str,
    tables: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
    marker_label: str,
) -> None:
    if not _table_exists(conn, table):
        return
    stats = _stats_bucket(tables, table)
    rows = conn.execute(f"SELECT {id_column}, {data_column} FROM {table}").fetchall()
    for row in rows:
        raw = str(row[data_column] or "")
        compacted = _clip_text(raw, limit=text_max_chars, marker_label=marker_label)
        _record_compaction(stats, totals, raw, compacted)
        if apply and compacted != raw:
            conn.execute(
                f"UPDATE {table} SET {data_column}=? WHERE {id_column}=?",
                (compacted, row[id_column]),
            )


def _compact_fact_value(key: str, raw: str, *, text_max_chars: int) -> str:
    try:
        payload = json.loads(raw)
    except Exception:
        return _clip_text(raw, limit=text_max_chars, marker_label="fact value")
    changed = False
    if isinstance(payload, dict) and key.startswith("durable_failure:"):
        payload = dict(payload)
        summary = payload.pop("last_summary", None)
        if summary:
            payload.update(_summary_preview_fields(summary, text_max_chars=min(1200, text_max_chars)))
            changed = True
    compacted = _compact_value(payload, text_max_chars=text_max_chars)
    if compacted == payload and not changed:
        return raw
    return json.dumps(compacted, ensure_ascii=False, sort_keys=True)


def _compact_json_text(raw: str, *, text_max_chars: int) -> str:
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        return _clip_text(raw, limit=text_max_chars, marker_label="json value")
    compacted = _compact_value(payload, text_max_chars=text_max_chars)
    if compacted == payload:
        return raw
    return json.dumps(compacted, ensure_ascii=False, sort_keys=True)


def _compact_chat_messages(
    conn: sqlite3.Connection,
    *,
    tables: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    if not _table_exists(conn, "chat_messages"):
        return
    stats = _stats_bucket(tables, "chat_messages")
    rows = conn.execute(
        "SELECT id, content FROM chat_messages WHERE status IN ('processed', 'delivered')"
    ).fetchall()
    for row in rows:
        raw = str(row["content"] or "")
        compacted = _clip_text(raw, limit=text_max_chars, marker_label="chat message")
        _record_compaction(stats, totals, raw, compacted)
        if apply and compacted != raw:
            conn.execute("UPDATE chat_messages SET content=? WHERE id=?", (compacted, row["id"]))


def _compact_meta_reflections(
    conn: sqlite3.Connection,
    *,
    tables: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    if not _table_exists(conn, "meta_reflections"):
        return
    stats = _stats_bucket(tables, "meta_reflections")
    rows = conn.execute(
        "SELECT id, diagnosis, proposal, verification_plan FROM meta_reflections"
    ).fetchall()
    for row in rows:
        updates: dict[str, str] = {}
        for column in ("diagnosis", "proposal", "verification_plan"):
            raw = str(row[column] or "")
            compacted = _clip_text(raw, limit=text_max_chars, marker_label=f"meta_reflection {column}")
            _record_compaction(stats, totals, raw, compacted)
            if compacted != raw:
                updates[column] = compacted
        if apply and updates:
            assignments = ", ".join(f"{column}=?" for column in updates)
            conn.execute(
                f"UPDATE meta_reflections SET {assignments} WHERE id=?",
                (*updates.values(), row["id"]),
            )


def _compact_value(value: Any, *, text_max_chars: int, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _clip_text(value, limit=text_max_chars, marker_label="persistent storage")
    if isinstance(value, dict):
        if depth >= 5:
            return _clip_text(_json_preview(value), limit=text_max_chars, marker_label="persistent storage")
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_COLLECTION_MAX_ITEMS]:
            compacted[str(key)] = _compact_value(item, text_max_chars=text_max_chars, depth=depth + 1)
        omitted = len(items) - len(compacted)
        if omitted > 0:
            compacted["_persistent_omitted_items"] = omitted
        return compacted
    if isinstance(value, (list, tuple)):
        if depth >= 5:
            return _clip_text(_json_preview(value), limit=text_max_chars, marker_label="persistent storage")
        items = list(value)
        if len(items) <= _COLLECTION_MAX_ITEMS:
            return [
                _compact_value(item, text_max_chars=text_max_chars, depth=depth + 1)
                for item in items
            ]
        retained_items = max(2, _COLLECTION_MAX_ITEMS - 1)
        head_count = max(1, retained_items // 2)
        tail_count = max(1, retained_items - head_count)
        omitted = len(items) - head_count - tail_count
        return [
            *[
                _compact_value(item, text_max_chars=text_max_chars, depth=depth + 1)
                for item in items[:head_count]
            ],
            {"_persistent_omitted_items": omitted},
            *[
                _compact_value(item, text_max_chars=text_max_chars, depth=depth + 1)
                for item in items[-tail_count:]
            ],
        ]
    return value


def _clip_text(text: Any, *, limit: int, marker_label: str) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    marker = f"\n...[{marker_label} truncated chars={len(value)} sha256={digest}]...\n"
    budget = max(0, limit - len(marker))
    head = max(200, budget // 2)
    tail = max(0, budget - head)
    return value[:head] + marker + (value[-tail:] if tail else "")


def _summary_preview_fields(summary: Any, *, text_max_chars: int) -> dict[str, Any]:
    text = str(summary or "")
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return {
        "last_summary_preview": _clip_text(text, limit=text_max_chars, marker_label="durable failure summary"),
        "last_summary_chars": len(text),
        "last_summary_sha256": digest,
    }


def _record_compaction(
    stats: dict[str, int],
    totals: dict[str, int],
    raw: str,
    compacted: str,
) -> None:
    original_bytes = len(raw.encode("utf-8", errors="replace"))
    compacted_bytes = len(compacted.encode("utf-8", errors="replace"))
    changed = compacted != raw
    stats["scanned_rows"] += 1
    totals["scanned_rows"] += 1
    if changed:
        stats["changed_rows"] += 1
        totals["changed_rows"] += 1
        stats["original_bytes"] += original_bytes
        stats["compacted_bytes"] += compacted_bytes
        totals["original_bytes"] += original_bytes
        totals["compacted_bytes"] += compacted_bytes


def _stats_bucket(tables: dict[str, dict[str, int]], table: str) -> dict[str, int]:
    return tables.setdefault(table, {
        "scanned_rows": 0,
        "changed_rows": 0,
        "original_bytes": 0,
        "compacted_bytes": 0,
    })


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _json_preview(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)
