"""Filesystem memory compaction for oversized semantic JSON artifacts."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .runtime_compaction import _clip_text, _compact_json_text, _compact_value, _table_exists

_TEXT_MAX_CHARS = 12000
_MEMORY_JSON_DIRS = ("nodes", "archive", "archive_by_prefix")
_EPISODIC_MARKDOWN_DIR = "episodic"


def compact_memory_dir(
    memory_dir: Path | str,
    *,
    apply: bool = False,
    vacuum: bool = False,
    text_max_chars: int = _TEXT_MAX_CHARS,
) -> dict[str, Any]:
    root = Path(memory_dir).expanduser()
    if not root.exists():
        return {
            "memory_dir": str(root),
            "dry_run": not apply,
            "error": "memory_dir_not_found",
            "scanned_files": 0,
            "changed_files": 0,
            "bad_json_files": 0,
            "original_bytes": 0,
            "compacted_bytes": 0,
            "saved_bytes": 0,
            "dirs": {},
            "dbs": {},
            "vacuumed": False,
        }

    dirs: dict[str, dict[str, int]] = {}
    dbs: dict[str, dict[str, int | bool]] = {}
    totals = {
        "scanned_files": 0,
        "changed_files": 0,
        "bad_json_files": 0,
        "original_bytes": 0,
        "compacted_bytes": 0,
        "scanned_rows": 0,
        "changed_rows": 0,
    }
    for dirname in _MEMORY_JSON_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.json")):
            _compact_json_file(
                path,
                bucket=dirname,
                dirs=dirs,
                totals=totals,
                apply=apply,
                text_max_chars=text_max_chars,
            )
        for path in sorted(base.rglob("*.jsonl")):
            _compact_jsonl_file(
                path,
                bucket=dirname,
                dirs=dirs,
                totals=totals,
                apply=apply,
                text_max_chars=text_max_chars,
            )

    _compact_episodic_markdown_files(
        root,
        dirs=dirs,
        totals=totals,
        apply=apply,
        text_max_chars=text_max_chars,
    )

    vacuumed = False
    for db_name in ("semantic.db", "episodic.db"):
        db_path = root / db_name
        if not db_path.exists():
            continue
        db_report = _compact_memory_db(
            db_path,
            apply=apply,
            vacuum=vacuum,
            text_max_chars=text_max_chars,
        )
        dbs[db_name] = db_report
        totals["scanned_rows"] += int(db_report.get("scanned_rows", 0))
        totals["changed_rows"] += int(db_report.get("changed_rows", 0))
        totals["original_bytes"] += int(db_report.get("original_bytes", 0))
        totals["compacted_bytes"] += int(db_report.get("compacted_bytes", 0))
        vacuumed = vacuumed or bool(db_report.get("vacuumed"))

    return {
        "memory_dir": str(root),
        "dry_run": not apply,
        "scanned_files": totals["scanned_files"],
        "changed_files": totals["changed_files"],
        "bad_json_files": totals["bad_json_files"],
        "scanned_rows": totals["scanned_rows"],
        "changed_rows": totals["changed_rows"],
        "original_bytes": totals["original_bytes"],
        "compacted_bytes": totals["compacted_bytes"],
        "saved_bytes": max(0, totals["original_bytes"] - totals["compacted_bytes"]),
        "dirs": dirs,
        "dbs": dbs,
        "vacuumed": vacuumed,
    }


def _compact_memory_db(
    db_path: Path,
    *,
    apply: bool,
    vacuum: bool,
    text_max_chars: int,
) -> dict[str, int | bool]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        totals = {"scanned_rows": 0, "changed_rows": 0, "original_bytes": 0, "compacted_bytes": 0}
        if db_path.name == "semantic.db":
            _compact_semantic_db(conn, totals=totals, apply=apply, text_max_chars=text_max_chars)
        elif db_path.name == "episodic.db":
            _compact_episodic_db(conn, totals=totals, apply=apply, text_max_chars=text_max_chars)

        vacuumed = False
        if apply:
            conn.commit()
            if vacuum:
                conn.execute("VACUUM")
                vacuumed = True
        else:
            conn.rollback()
        return {**totals, "vacuumed": vacuumed}
    finally:
        conn.close()


def _compact_semantic_db(
    conn: sqlite3.Connection,
    *,
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    if not _table_exists(conn, "nodes"):
        return
    rows = conn.execute("SELECT id, title, body, tags FROM nodes").fetchall()
    for row in rows:
        raw_title = str(row["title"] or "")
        raw_body = str(row["body"] or "")
        title = _clip_text(raw_title, limit=min(1000, text_max_chars), marker_label="semantic title")
        body = _clip_text(raw_body, limit=text_max_chars, marker_label="semantic body")
        _record_db_text(totals, raw_title, title)
        _record_db_text(totals, raw_body, body)
        if apply and (title != raw_title or body != raw_body):
            conn.execute("UPDATE nodes SET title=?, body=? WHERE id=?", (title, body, row["id"]))
            _sync_semantic_fts(conn, node_id=str(row["id"]), title=title, body=body, tags=str(row["tags"] or "[]"))


def _compact_episodic_db(
    conn: sqlite3.Connection,
    *,
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    if _table_exists(conn, "events"):
        rows = conn.execute("SELECT id, data FROM events").fetchall()
        for row in rows:
            raw = str(row["data"] or "")
            compacted = _compact_json_text(raw, text_max_chars=text_max_chars)
            _record_db_text(totals, raw, compacted)
            if apply and compacted != raw:
                conn.execute("UPDATE events SET data=? WHERE id=?", (compacted, row["id"]))
    if _table_exists(conn, "narrative"):
        rows = conn.execute("SELECT id, task_id, chat_id, interlocutor_id, role, content FROM narrative").fetchall()
        for row in rows:
            raw = str(row["content"] or "")
            compacted = _clip_text(raw, limit=text_max_chars, marker_label="episodic narrative")
            _record_db_text(totals, raw, compacted)
            if apply and compacted != raw:
                conn.execute("UPDATE narrative SET content=? WHERE id=?", (compacted, row["id"]))
                _sync_narrative_fts(conn, row=row, content=compacted)


def _sync_semantic_fts(conn: sqlite3.Connection, *, node_id: str, title: str, body: str, tags: str) -> None:
    if not _table_exists(conn, "nodes_fts"):
        return
    conn.execute("DELETE FROM nodes_fts WHERE id=?", (node_id,))
    conn.execute(
        "INSERT INTO nodes_fts(id, title, body, tags) VALUES (?, ?, ?, ?)",
        (node_id, title, body, tags),
    )


def _sync_narrative_fts(conn: sqlite3.Connection, *, row: sqlite3.Row, content: str) -> None:
    if not _table_exists(conn, "narrative_fts"):
        return
    conn.execute("DELETE FROM narrative_fts WHERE id=?", (row["id"],))
    conn.execute(
        "INSERT INTO narrative_fts(id, task_id, chat_id, interlocutor_id, role, content) VALUES (?, ?, ?, ?, ?, ?)",
        (
            row["id"],
            str(row["task_id"] or ""),
            str(row["chat_id"] or ""),
            str(row["interlocutor_id"] or ""),
            str(row["role"] or ""),
            content,
        ),
    )


def _record_db_text(totals: dict[str, int], raw: str, compacted: str) -> None:
    totals["scanned_rows"] += 1
    if compacted == raw:
        return
    totals["changed_rows"] += 1
    totals["original_bytes"] += len(raw.encode("utf-8", errors="replace"))
    totals["compacted_bytes"] += len(compacted.encode("utf-8", errors="replace"))


def _compact_json_file(
    path: Path,
    *,
    bucket: str,
    dirs: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    stats = _stats_bucket(dirs, bucket)
    totals["scanned_files"] += 1
    stats["scanned_files"] += 1
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw or "{}")
    except Exception:
        totals["bad_json_files"] += 1
        stats["bad_json_files"] += 1
        return
    compacted = _compact_value(payload, text_max_chars=text_max_chars)
    if compacted == payload:
        return
    compacted_text = json.dumps(compacted, ensure_ascii=False, indent=2, sort_keys=True)
    original_bytes = len(raw.encode("utf-8", errors="replace"))
    compacted_bytes = len(compacted_text.encode("utf-8", errors="replace"))
    totals["changed_files"] += 1
    stats["changed_files"] += 1
    totals["original_bytes"] += original_bytes
    totals["compacted_bytes"] += compacted_bytes
    stats["original_bytes"] += original_bytes
    stats["compacted_bytes"] += compacted_bytes
    if apply:
        path.write_text(compacted_text, encoding="utf-8")


def _compact_jsonl_file(
    path: Path,
    *,
    bucket: str,
    dirs: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    stats = _stats_bucket(dirs, bucket)
    totals["scanned_files"] += 1
    stats["scanned_files"] += 1
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    compacted_lines: list[str] = []
    changed = False
    bad_record = False
    for line in lines:
        if not line.strip():
            compacted_lines.append(line)
            continue
        try:
            payload = json.loads(line)
        except Exception:
            bad_record = True
            compacted_lines.append(line)
            continue
        compacted = _compact_value(payload, text_max_chars=text_max_chars)
        if compacted != payload:
            changed = True
        compacted_lines.append(json.dumps(compacted, ensure_ascii=False, sort_keys=True))
    if bad_record:
        totals["bad_json_files"] += 1
        stats["bad_json_files"] += 1
    if not changed:
        return
    compacted_text = "\n".join(compacted_lines)
    if raw.endswith("\n"):
        compacted_text += "\n"
    original_bytes = len(raw.encode("utf-8", errors="replace"))
    compacted_bytes = len(compacted_text.encode("utf-8", errors="replace"))
    totals["changed_files"] += 1
    stats["changed_files"] += 1
    totals["original_bytes"] += original_bytes
    totals["compacted_bytes"] += compacted_bytes
    stats["original_bytes"] += original_bytes
    stats["compacted_bytes"] += compacted_bytes
    if apply:
        path.write_text(compacted_text, encoding="utf-8")


def _compact_episodic_markdown_files(
    root: Path,
    *,
    dirs: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    base = root / _EPISODIC_MARKDOWN_DIR
    if not base.exists():
        return
    for path in sorted(base.rglob("*.md")):
        _compact_text_file(
            path,
            bucket=_EPISODIC_MARKDOWN_DIR,
            dirs=dirs,
            totals=totals,
            apply=apply,
            text_max_chars=text_max_chars,
        )


def _compact_text_file(
    path: Path,
    *,
    bucket: str,
    dirs: dict[str, dict[str, int]],
    totals: dict[str, int],
    apply: bool,
    text_max_chars: int,
) -> None:
    stats = _stats_bucket(dirs, bucket)
    totals["scanned_files"] += 1
    stats["scanned_files"] += 1
    raw = path.read_text(encoding="utf-8")
    compacted = _clip_text(raw, limit=text_max_chars, marker_label="episodic markdown")
    if compacted == raw:
        return
    original_bytes = len(raw.encode("utf-8", errors="replace"))
    compacted_bytes = len(compacted.encode("utf-8", errors="replace"))
    totals["changed_files"] += 1
    stats["changed_files"] += 1
    totals["original_bytes"] += original_bytes
    totals["compacted_bytes"] += compacted_bytes
    stats["original_bytes"] += original_bytes
    stats["compacted_bytes"] += compacted_bytes
    if apply:
        path.write_text(compacted, encoding="utf-8")


def _stats_bucket(dirs: dict[str, dict[str, int]], dirname: str) -> dict[str, int]:
    return dirs.setdefault(dirname, {
        "scanned_files": 0,
        "changed_files": 0,
        "bad_json_files": 0,
        "original_bytes": 0,
        "compacted_bytes": 0,
    })
