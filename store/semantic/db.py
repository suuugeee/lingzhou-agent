from __future__ import annotations

import json
import logging as _log_sem
import sqlite3
import struct
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from typing import Any

from . import _DDL, _DDL_EMBEDDINGS, _DDL_FTS5, MemoryNode, _parse_table_cols

_log = _log_sem.getLogger("lingzhou.memory.semantic")

# body 存储上限：超过此值截取末尾（保留最近内容），避免历史累积无限增长导致 OOM
_BODY_STORE_MAX = 2 * 1024 * 1024   # 2 MB
# FTS 分词上限：FTS5 不需要全文，只需前 256KB 即可支持关键词检索
_FTS_BODY_MAX = 256 * 1024           # 256 KB


def _vec_to_blob(vec: list[float]) -> bytes:
    """float32 列表 → 4 bytes/dim BLOB（与 sqlite-vec / pgvector 惯例一致）。"""
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    """4 bytes/dim BLOB → float32 列表；兼容旧 JSON TEXT 回退。"""
    if isinstance(blob, (bytes, bytearray, memoryview)):
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", bytes(blob)[:n * 4]))
    # 旧格式兼容：JSON TEXT
    if isinstance(blob, str):
        return json.loads(blob)
    return blob  # 已是 list


def _normalize_interlocutor_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = str(raw or "").strip()
        if not tag:
            continue
        if tag == "person_profile":
            tag = "interlocutor_profile"
        elif tag.startswith("person:"):
            tag = "interlocutor:" + tag.split(":", 1)[1]
        if tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def _is_legacy_interlocutor_profile(cls, node: MemoryNode) -> bool:
    tags = {str(tag or "").strip() for tag in (node.tags or [])}
    return bool(
        node.kind == "person"
        and (
            getattr(node, "source", "") in {"user_profile", "person_profile"}
            or "person_profile" in tags
            or any(tag.startswith("person:") for tag in tags)
            or any(tag.startswith("handle:") for tag in tags)
        )
    )


def _migrate_interlocutor_profiles(self, max_seconds: float | None = None) -> None:
    started = time.monotonic()
    migrated = 0
    scanned = 0
    for path in self._dir.glob("*.json"):
        scanned += 1
        if max_seconds is not None and max_seconds > 0 and time.monotonic() - started >= max_seconds:
            self._maintenance_deferred = True
            _log.info(
                "[semantic] 旧画像迁移超过启动预算 %.1fs，已延期 scanned=%d migrated=%d",
                max_seconds,
                scanned,
                migrated,
            )
            break
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            node = MemoryNode.from_dict(payload)
        except Exception:
            continue
        if not self._is_legacy_interlocutor_profile(node):
            continue

        new_tags = self._normalize_interlocutor_tags(list(node.tags or []))
        new_source = "interlocutor_profile" if (node.source or "") in {"", "user_profile", "person_profile"} else node.source
        changed = bool(node.kind != "interlocutor" or new_tags != list(node.tags or []) or new_source != node.source)
        if not changed:
            continue

        migrated += 1
        node.kind = "interlocutor"
        node.tags = new_tags
        node.source = new_source
        path.write_text(json.dumps(node.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._db_upsert(node)

    if migrated:
        _log.info("[semantic] 已迁移 %d 个旧 person_profile 节点到 interlocutor_profile", migrated)


def _conn_getter(self) -> sqlite3.Connection:
    conn = getattr(self, "_conn_ref", None)
    if conn is None:
        raise RuntimeError("semantic db session is not open")
    return conn


def _conn_setter(self, value: sqlite3.Connection | None) -> None:
    self._conn_ref = value


@contextmanager
def _db_session(self):
    with self._db_lock:
        if self._conn_ref is not None:
            self._session_depth += 1
            try:
                yield self._conn_ref
            finally:
                self._session_depth -= 1
            return

        conn = self._open_db()
        self._conn = conn
        self._session_depth = 1
        try:
            yield conn
        finally:
            self._session_depth -= 1
            if self._session_depth == 0:
                self.close()


def close(self) -> None:
    conn = getattr(self, "_conn_ref", None)
    self._conn_ref = None
    self._session_depth = 0
    if conn is not None:
        with suppress(Exception):
            conn.close()


def _setup_embeddings_table(self, conn: sqlite3.Connection) -> None:
    """创建多模态 embedding 表（幂等，已存在则跳过）。"""
    try:
        conn.executescript(_DDL_EMBEDDINGS)
        conn.commit()
    except Exception as exc:
        _log.warning("[semantic] node_embeddings 表初始化失败: %s", exc)


def _migrate_embeddings(self, batch_limit: int = 500) -> None:
    """将 nodes.embedding 历史数据（一次性幂等）迁移到 node_embeddings 表。"""
    try:
        rows = self._conn.execute(
            """
            SELECT n.id, n.embedding, n.created_at
            FROM nodes n
            WHERE n.embedding IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM node_embeddings e
                  WHERE e.node_id = n.id AND e.modality = 'text' AND e.model = 'legacy'
              )
            LIMIT ?
            """,
            (max(1, int(batch_limit)),),
        ).fetchall()
        if not rows:
            return
        now = datetime.now(UTC).isoformat()
        count = 0
        for row in rows:
            node_id, emb_raw, created_at = row[0], row[1], row[2]
            if not emb_raw:
                continue
            try:
                # 旧列是 JSON TEXT，转换为 float32 BLOB
                vec = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
                blob = _vec_to_blob(vec)
                self._conn.execute(
                    "INSERT OR IGNORE INTO node_embeddings"
                    " (node_id, modality, model, dim, vector, created_at)"
                    " VALUES (?, 'text', 'legacy', ?, ?, ?)",
                    (node_id, len(vec), blob, created_at or now),
                )
                count += 1
            except Exception:
                pass
        self._conn.commit()
        if count:
            _log.info("[semantic] 已迁移 %d 个旧 embedding 到 node_embeddings", count)
        if len(rows) >= max(1, int(batch_limit)):
            _log.info("[semantic] 旧 embedding 迁移达到启动批量上限 %d，剩余部分延期", batch_limit)
    except Exception as exc:
        _log.warning("[semantic] embedding 迁移失败，跳过: %s", exc)


def _open_db(self) -> sqlite3.Connection:
    try:
        conn = self._connect()
        conn.executescript(_DDL)
        conn.commit()
        self._setup_fts5(conn)
        self._setup_embeddings_table(conn)
        return conn
    except sqlite3.DatabaseError:
        self._db_path.unlink(missing_ok=True)
        conn = self._connect()
        conn.executescript(_DDL)
        conn.commit()
        self._setup_fts5(conn)
        self._setup_embeddings_table(conn)
        return conn


def _migrate(self) -> None:
    try:
        desired = _parse_table_cols(_DDL)
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(nodes)")}
        changed = False
        for col, definition in desired.items():
            if col not in existing:
                self._conn.execute(f"ALTER TABLE nodes ADD COLUMN {col} {definition}")
                changed = True
        if changed:
            self._conn.commit()
    except Exception:
        pass
    try:
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind)"
        )
        self._conn.commit()
    except Exception:
        pass
    self._migrate_embeddings()


def _setup_fts5(self, conn: sqlite3.Connection) -> None:
    try:
        try:
            conn.execute("SELECT id FROM nodes_fts LIMIT 0")
        except Exception:
            _log.warning("[semantic] nodes_fts 缺少 id 列，重建 FTS5 表")
            conn.execute("DROP TABLE IF EXISTS nodes_fts")
            conn.commit()
        conn.executescript(_DDL_FTS5)
        conn.commit()
        self._fts5_ok = True
    except Exception as exc:
        _log.warning("[semantic] FTS5 初始化失败，降级为全表扫描：%s", exc)
        self._fts5_ok = False


def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(str(self._db_path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _sync_from_files(self, max_seconds: float | None = None) -> None:
    try:
        started = time.monotonic()
        existing_ids: set[str] = {
            row[0] for row in self._conn.execute("SELECT id FROM nodes")
        }
        scanned = 0
        imported = 0
        for p in self._dir.glob("*.json"):
            scanned += 1
            if max_seconds is not None and max_seconds > 0 and time.monotonic() - started >= max_seconds:
                self._maintenance_deferred = True
                _log.info(
                    "[semantic] JSON→索引同步超过启动预算 %.1fs，已延期 scanned=%d imported=%d",
                    max_seconds,
                    scanned,
                    imported,
                )
                break
            try:
                if p.stem in existing_ids:
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("id") not in existing_ids:
                    self._db_upsert(MemoryNode.from_dict(d))
                    imported += 1
            except Exception as exc:
                _log.warning("[semantic] 跳过损坏的节点文件 %s: %s", p.name, exc)
        self._conn.commit()
    except Exception as exc:
        _log.warning("[semantic] _sync_from_files 失败，回退到文件扫描: %s", exc)


def _validate_and_repair_index(self) -> None:
    try:
        db_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        has_json = next(self._dir.glob("*.json"), None) is not None
        if has_json and (db_count == 0 or not self._fts5_ok):
            self._maintenance_deferred = True
            _log.warning(
                "[semantic] 索引为空或 FTS5 异常 (has_json=%s, db=%d, fts5=%s)，"
                "启动期跳过全量重建；运行可继续，必要时手动 rebuild_index",
                has_json,
                db_count,
                self._fts5_ok,
            )
    except Exception as exc:
        _log.warning("[semantic] 索引校验失败，跳过自动重建: %s", exc)


def rebuild_index(self) -> None:
    with self._db_session():
        self._conn.execute("DELETE FROM nodes")
        if self._fts5_ok:
            with suppress(Exception):
                self._conn.execute("DELETE FROM nodes_fts")
        with suppress(Exception):
            self._conn.execute("DELETE FROM node_embeddings")
        self._conn.commit()
        for p in self._dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                self._db_upsert(MemoryNode.from_dict(d))
                emb = d.get("embedding")
                if emb is not None:
                    vec = json.loads(emb) if isinstance(emb, str) else emb
                    blob = _vec_to_blob(vec)
                    self._conn.execute(
                        "INSERT OR IGNORE INTO node_embeddings"
                        " (node_id, modality, model, dim, vector, created_at)"
                        " VALUES (?, 'text', 'legacy', ?, ?, ?)",
                        (d.get("id"), len(vec), blob,
                         d.get("created_at") or datetime.now(UTC).isoformat()),
                    )
            except Exception:
                pass
        self._conn.commit()


def _db_upsert(self, node: MemoryNode) -> None:
    tags_json = json.dumps(node.tags, ensure_ascii=False)
    body = _stored_node_body(node)
    self._conn.execute(
        """INSERT INTO nodes
                         (id, kind, title, body, activation, valence, importance, tags, source, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             kind=excluded.kind,
             title=excluded.title,
             body=excluded.body,
             activation=excluded.activation,
             valence=excluded.valence,
                             importance=excluded.importance,
             tags=excluded.tags,
             source=excluded.source""",
        (
            node.id, node.kind, node.title, body,
            node.activation, node.valence,
                            node.importance,
            tags_json,
            getattr(node, "source", ""),
            node.created_at,
        ),
    )
    self._conn.commit()
    if self._fts5_ok:
        try:
            self._sync_node_fts(
                node_id=node.id,
                title=node.title,
                body=body,
                tags_json=tags_json,
            )
        except Exception as exc:
            with suppress(Exception):
                self._conn.rollback()
            self._fts5_ok = False
            _log.warning("[semantic] FTS5 同步失败，降级为全表扫描: %s", exc)


def _sync_node_fts(
    self,
    *,
    node_id: str,
    title: str,
    body: str,
    tags_json: str,
) -> None:
    body_fts = body[:_FTS_BODY_MAX] if body and len(body) > _FTS_BODY_MAX else body
    self._conn.execute("DELETE FROM nodes_fts WHERE id = ?", (node_id,))
    self._conn.execute(
        "INSERT INTO nodes_fts(id, title, body, tags) VALUES (?, ?, ?, ?)",
        (node_id, title, body_fts, tags_json),
    )
    self._conn.commit()


def fts5_ok(self) -> bool:
    return self._fts5_ok


def decay_lambda(self) -> float:
    return self._decay_lambda


def stats(self) -> dict[str, Any]:
    total_nodes = 0
    with self._db_session():
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
            total_nodes = int(row[0] or 0) if row else 0
        except Exception:
            total_nodes = 0
    maintenance = {}
    try:
        maintenance = self._maintenance.snapshot()
    except Exception:
        maintenance = {}
    return {
        "nodes": total_nodes,
        "fts5_ok": bool(self._fts5_ok),
        "decay_lambda": float(self._decay_lambda),
        "embedding_enabled": bool(self._embed_fn is not None),
        "source_weight": float(self._source_weight),
        "temporal_weight": float(self._temporal_weight),
        "temporal_window_days": float(self._temporal_window_days),
        "db_path": str(self._db_path),
        "nodes_dir": str(self._dir),
        "maintenance_state": maintenance.get("state", "unknown"),
        "maintenance_deferred": bool(maintenance.get("deferred", False)),
        "maintenance_last_error": str(maintenance.get("last_error", "")),
        "maintenance_last_startup_seconds": float(maintenance.get("last_startup_seconds") or 0.0),
        "maintenance_last_background_seconds": float(maintenance.get("last_background_seconds") or 0.0),
    }


def upsert(self, node: MemoryNode) -> None:
    stored_node = _node_for_storage(node)
    path = self._dir / f"{node.id}.json"
    path.write_text(json.dumps(stored_node.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    with self._db_session():
        try:
            self._db_upsert(stored_node)
        except Exception as exc:
            _log.warning("[semantic] 节点写入 DB 失败，保留 json 作为恢复源: %s", exc)

        # 与“按需向量化”策略一致，避免在检索前无条件预计算 embedding。
        # 仅在外部显式调用 set_embedding 或节点文件已携带 legacy embedding 时再持久化。
        legacy_embedding = getattr(node, "embedding", None)
        if legacy_embedding is not None:
            try:
                vec = json.loads(legacy_embedding) if isinstance(legacy_embedding, str) else legacy_embedding
                blob = _vec_to_blob(vec)
                self._conn.execute(
                    "INSERT OR REPLACE INTO node_embeddings"
                    " (node_id, modality, model, dim, vector, created_at)"
                    " VALUES (?, 'text', 'legacy', ?, ?, ?)",
                    (node.id, len(vec), blob, datetime.now(UTC).isoformat()),
                )
                self._conn.commit()
            except Exception:
                pass


def _stored_node_body(node: MemoryNode) -> str:
    body = node.body or ""
    if body and len(body) > _BODY_STORE_MAX:
        body = body[-_BODY_STORE_MAX:]  # 保留末尾（最新内容），丢弃远古历史
        _log.warning(
            "[semantic] body 超过 %d bytes，已截断至最近 %d bytes: id=%s kind=%s",
            _BODY_STORE_MAX, _BODY_STORE_MAX, node.id, node.kind,
        )
    return body


def _node_for_storage(node: MemoryNode) -> MemoryNode:
    body = _stored_node_body(node)
    if body == (node.body or ""):
        return node
    return MemoryNode(
        id=node.id,
        kind=node.kind,
        title=node.title,
        body=body,
        activation=node.activation,
        valence=node.valence,
        importance=node.importance,
        tags=list(node.tags or []),
        source=node.source,
        created_at=node.created_at,
    )


def get(self, node_id: str) -> MemoryNode | None:
    with self._db_session():
        try:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row:
                return self._row_to_node(row)
        except Exception:
            pass
    path = self._dir / f"{node_id}.json"
    if path.exists():
        return MemoryNode.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return None


def find_by_title(self, title: str, limit: int = 10) -> list[MemoryNode]:
    normalized = (title or "").strip()
    if not normalized:
        return []
    with self._db_session():
        try:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE title = ? ORDER BY created_at DESC LIMIT ?",
                (normalized, limit),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]
        except Exception:
            pass
    hits: list[MemoryNode] = []
    for p in self._dir.glob("*.json"):
        try:
            node = MemoryNode.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
        if node.title == normalized:
            hits.append(node)
            if len(hits) >= limit:
                break
    hits.sort(key=lambda item: item.created_at, reverse=True)
    return hits[:limit]
