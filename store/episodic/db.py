from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from store.compact import compact_runtime_mapping

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- 结构化事件表（替代 events.jsonl O(n) 扫描）
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    ts         TEXT NOT NULL,
    data       TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_type_id ON events(event_type, id DESC);

-- 叙事记录表（.md 文件的结构化镜像，用于 FTS5 检索）
CREATE TABLE IF NOT EXISTS narrative (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT,
    chat_id     TEXT,
    interlocutor_id TEXT,
    role        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    content     TEXT NOT NULL,
    affect      TEXT,
    ts          TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS narrative_fts USING fts5(
    id UNINDEXED,
    task_id,
    chat_id,
    interlocutor_id,
    role,
    content,
    tokenize='unicode61'
);
-- 查询索引（幂等，DDL 统一管理）
CREATE INDEX IF NOT EXISTS idx_narrative_task_id ON narrative(task_id);
CREATE INDEX IF NOT EXISTS idx_narrative_chat_id ON narrative(chat_id);
CREATE INDEX IF NOT EXISTS idx_narrative_interlocutor_id ON narrative(interlocutor_id);
CREATE INDEX IF NOT EXISTS idx_narrative_ts ON narrative(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def open_db(memory) -> sqlite3.Connection:
    """打开 DB；损坏时自动删除并重建（migrate_from_jsonl 重新导入历史）。"""
    try:
        conn = connect(memory)
        conn.executescript(DDL)
        ensure_schema_compat(memory, conn)
        conn.commit()
        return conn
    except sqlite3.DatabaseError:
        memory._db_path.unlink(missing_ok=True)
        conn = connect(memory)
        conn.executescript(DDL)
        ensure_schema_compat(memory, conn)
        conn.commit()
        return conn


def connect(memory) -> sqlite3.Connection:
    # timeout=60: C 层等待 60s；WAL + busy_timeout 防止多线程/多进程写冲突
    conn = sqlite3.connect(str(memory._db_path), check_same_thread=False, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema_compat(memory, conn: sqlite3.Connection) -> None:
    narrative_columns = set(table_columns(conn, "narrative"))
    if "chat_id" not in narrative_columns:
        conn.execute("ALTER TABLE narrative ADD COLUMN chat_id TEXT")
    if "interlocutor_id" not in narrative_columns:
        conn.execute("ALTER TABLE narrative ADD COLUMN interlocutor_id TEXT")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_narrative_chat_id ON narrative(chat_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_narrative_interlocutor_id ON narrative(interlocutor_id)")

    fts_columns = set(table_columns(conn, "narrative_fts"))
    if "chat_id" not in fts_columns or "interlocutor_id" not in fts_columns:
        rebuild_narrative_fts(memory, conn)



def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    except Exception:
        return []
    return [str(row[1]) for row in rows]


def rebuild_narrative_fts(memory, conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS narrative_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE narrative_fts USING fts5("
        " id UNINDEXED,"
        " task_id,"
        " chat_id,"
        " interlocutor_id,"
        " role,"
        " content,"
        " tokenize='unicode61'"
        ")"
    )
    conn.execute(
        "INSERT INTO narrative_fts(id, task_id, chat_id, interlocutor_id, role, content) "
        "SELECT id, COALESCE(task_id, ''), COALESCE(chat_id, ''), COALESCE(interlocutor_id, ''), role, content FROM narrative"
    )


def migrate_from_jsonl(memory) -> None:
    """一次性：将历史 events.jsonl 导入 SQLite events 表（幂等，count>0 时跳过）。"""
    path = memory._dir / "events.jsonl"
    if not path.exists():
        return
    try:
        count = memory._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if count > 0:
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                et = data.pop("t", "unknown")
                ts = data.pop("ts", datetime.now(UTC).isoformat())
                memory._conn.execute(
                    "INSERT INTO events(event_type, ts, data) VALUES (?, ?, ?)",
                    (et, ts, json.dumps(compact_runtime_mapping(data), ensure_ascii=False)),
                )
            except Exception:
                pass
        memory._conn.commit()
    except Exception:
        pass
