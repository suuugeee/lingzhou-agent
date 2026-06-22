from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from store.compact import compact_runtime_mapping


def record_event(memory, event_type: str, data: dict[str, Any]) -> None:
    """追加一条结构化事件（perception / emotion 快照）。"""
    ts = datetime.now(UTC).isoformat()
    compacted_data = compact_runtime_mapping(data)
    try:
        with memory._db_session():
            memory._conn.execute(
                "INSERT INTO events(event_type, ts, data) VALUES (?, ?, ?)",
                (event_type, ts, json.dumps(compacted_data, ensure_ascii=False)),
            )
            memory._conn.commit()
            if memory._max_events > 0:
                rotate_events_db(memory, event_type)
    except Exception:
        path = memory._dir / "events.jsonl"
        entry: dict[str, Any] = {"t": event_type, "ts": ts, **compacted_data}
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


def rotate_events_db(memory, event_type: str) -> None:
    """保留该类型最新 max_events 条，删除超出的旧记录。"""
    try:
        memory._conn.execute(
            """DELETE FROM events WHERE event_type = ? AND id NOT IN (
                SELECT id FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?
            )""",
            (event_type, event_type, memory._max_events),
        )
        memory._conn.commit()
    except Exception:
        pass


def list_events(memory, event_type: str, limit: int = 10) -> list[dict[str, Any]]:
    """返回最近 limit 条指定类型事件（时间升序）。O(log n) 索引扫描。"""
    with memory._db_session():
        try:
            rows = memory._conn.execute(
                "SELECT ts, data FROM events WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in reversed(rows):
                try:
                    row_data = json.loads(row["data"])
                    row_data["t"] = event_type
                    row_data["ts"] = row["ts"]
                    result.append(row_data)
                except Exception:
                    pass
            return result
        except Exception:
            return fallback_list_events(memory, event_type, limit)


def fallback_list_events(memory, event_type: str, limit: int) -> list[dict[str, Any]]:
    """DB 不可用时回退到 JSONL 逆序扫描。"""
    path = memory._dir / "events.jsonl"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    matched: list[dict[str, Any]] = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            if data.get("t") == event_type:
                matched.append(data)
                if len(matched) >= limit:
                    break
        except Exception:
            pass
    matched.reverse()
    return matched


def list_events_multi(
    memory, event_types: list[str], limit: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """一次查询，按类型分桶返回最近 limit 条（时间升序）。O(log n) 索引扫描。"""
    result: dict[str, list[dict[str, Any]]] = {event_type: [] for event_type in event_types}
    if not event_types:
        return result

    with memory._db_session():
        try:
            placeholders = ",".join("?" * len(event_types))
            rows = memory._conn.execute(
                f"SELECT event_type, ts, data FROM events"
                f" WHERE event_type IN ({placeholders}) ORDER BY id DESC",
                event_types,
            ).fetchall()
            for row in rows:
                et = row["event_type"]
                if et in result and len(result[et]) < limit:
                    try:
                        row_data = json.loads(row["data"])
                        row_data["t"] = et
                        row_data["ts"] = row["ts"]
                        result[et].append(row_data)
                    except Exception:
                        pass
            for values in result.values():
                values.reverse()
            return result
        except Exception:
            path = memory._dir / "events.jsonl"
            if not path.exists():
                return result
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return result
            for raw in reversed(lines):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    event_type = data.get("t")
                    if event_type in result and len(result[event_type]) < limit:
                        result[event_type].append(data)
                except Exception:
                    pass
                if all(len(values) >= limit for values in result.values()):
                    break
            for values in result.values():
                values.reverse()
            return result
