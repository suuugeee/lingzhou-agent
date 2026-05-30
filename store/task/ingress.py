from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .chat import build_chat_message_insert
from .fact import build_fact_upsert
from .state import build_task_data, build_task_insert


class _IngressBase:
    """sqlite3 连接基础设施（内部共享）。"""

    _LOCKS: dict[str, threading.RLock] = {}
    _LOCKS_GUARD = threading.Lock()

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path).expanduser()

    def _db_lock(self) -> threading.RLock:
        key = str(self._path.resolve())
        with self._LOCKS_GUARD:
            lock = self._LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                self._LOCKS[key] = lock
            return lock

    def _write_with_retry(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                with self._db_lock():
                    return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "database is locked" in str(exc).lower() and attempt < 5:
                    time.sleep(0.15 * (2 ** attempt))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return None

    def _connect(self) -> sqlite3.Connection:
        # timeout=60: C 层等待 60s（对跨连接 SQLITE_BUSY 有效）
        # WAL 允许读写并发，减少写写冲突窗口
        conn = sqlite3.connect(str(self._path), timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn


class IngressWriter(_IngressBase):
    """只写入口：channel/webhook 线程侧写入操作。

    并发说明
    --------
    本类（及其子类 IngressStore）使用同步 sqlite3，供线程侧（channel/webhook）
    写入任务和消息。Runtime 内部的读写通过独立的 aiosqlite 异步路径
    git push cnb HEAD:bot --force    （store/task/_base.py + TaskStore 等）完成。两条路径共享同一个 SQLite 文件。

    SQLite WAL 模式允许读写并发，但写写仍然串行。同一进程内两个连接并发写会触发
    SQLITE_LOCKED（不同于跨进程的 SQLITE_BUSY），且不受 sqlite3.connect(timeout)
    控制。因此两条路径均通过 PRAGMA busy_timeout 在 SQLite 层设置等待时间，
    避免并发写时立即报 "database is locked"。不应在同一调用栈混用两条路径。
    """

    def add_chat_message(
        self,
        role: str,
        content: str,
        *,
        chat_id: str = "",
        status: str = "pending",
    ) -> int:
        def _do() -> int:
            insert_args = build_chat_message_insert(
                role,
                content,
                chat_id=chat_id,
                status=status,
            )
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO chat_messages(role, content, session_id, status) VALUES (?,?,?,?)",
                    insert_args,
                )
                return int(cur.lastrowid or 0)

        return self._write_with_retry(_do)

    def set_fact(self, key: str, value: str, *, scope: str = "general") -> None:
        def _do() -> None:
            sql, params = build_fact_upsert(key, value, scope=scope)
            with self._connect() as conn:
                conn.execute(sql, params)

        self._write_with_retry(_do)

    def ingest_user_message(
        self,
        content: str,
        *,
        chat_id: str,
        facts: dict[str, str | tuple[str, str]] | None = None,
    ) -> int:
        def _do() -> int:
            insert_args = build_chat_message_insert(
                "user",
                content,
                chat_id=chat_id,
                status="pending",
            )
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO chat_messages(role, content, session_id, status) VALUES (?,?,?,?)",
                    insert_args,
                )
                message_id = int(cur.lastrowid or 0)
                for key, raw_value in (facts or {}).items():
                    if isinstance(raw_value, tuple):
                        value, scope = raw_value
                    else:
                        value, scope = str(raw_value), "general"
                    sql, params = build_fact_upsert(key, value, scope=scope)
                    conn.execute(sql, params)
                return message_id

        return self._write_with_retry(_do)

    def mark_chat_message_delivered(self, message_id: int) -> None:
        def _do() -> None:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE chat_messages SET status='delivered' WHERE id=?",
                    (int(message_id),),
                )

        self._write_with_retry(_do)

    def add_task(
        self,
        title: str,
        *,
        goal: str = "",
        priority: str = "normal",
        source: str = "external",
        status: str = "pending",
        next_step: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int:
        data = build_task_data(
            goal=goal,
            source=source,
            next_step=next_step,
            extras=extras,
        )
        insert_args = build_task_insert(
            title,
            status=status,
            priority=priority,
            data=data,
        )
        def _do() -> int:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO tasks (title, status, priority, data) VALUES (?,?,?,?)",
                    insert_args,
                )
                return int(cur.lastrowid or 0)

        return self._write_with_retry(_do)


class IngressStore(IngressWriter):
    """读写完整入口（向后兼容）：在 IngressWriter 写能力基础上追加读查询。

    channel 侧同时需要读写时使用此类；纯写入侧可改用 IngressWriter。
    """

    def get_fact(self, key: str) -> tuple[str, bool]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        if row is None:
            return "", False
        return str(row[0] or ""), True

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def list_pending_assistant_messages(
        self,
        *,
        chat_prefix: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...]
        if chat_prefix:
            sql = (
                "SELECT id, content, session_id AS chat_id, created_at FROM chat_messages "
                "WHERE role='assistant' AND session_id LIKE ? "
                "AND status IN ('pending','processed') "
                "ORDER BY id ASC LIMIT ?"
            )
            params = (f"{chat_prefix}%", limit)
        else:
            sql = (
                "SELECT id, content, session_id AS chat_id, created_at FROM chat_messages "
                "WHERE role='assistant' AND status IN ('pending','processed') "
                "ORDER BY id ASC LIMIT ?"
            )
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "content": str(row["content"] or ""),
                "chat_id": str(row["chat_id"] or ""),
                "created_at": str(row["created_at"] or ""),
            }
            for row in rows
        ]
