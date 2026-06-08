from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .source import source_from_role

_log = logging.getLogger("lingzhou.episodic")

_DAILY_SEARCH_MAX_HITS = 5
_DAILY_SEARCH_EXCERPT_CHARS = 480


def narrative_filename(task_id: str | None) -> str:
    return f"task-{task_id}.md" if task_id else "global.md"


def chat_filename(chat_id: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(chat_id or "")).strip("._-")
    slug = normalized[:48] or "chat"
    digest = hashlib.md5(str(chat_id).encode("utf-8")).hexdigest()[:10]
    return f"chat-{slug}-{digest}.md"


def interlocutor_filename(interlocutor_id: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(interlocutor_id or "")).strip("._-")
    slug = normalized[:48] or "interlocutor"
    digest = hashlib.md5(str(interlocutor_id).encode("utf-8")).hexdigest()[:10]
    return f"interlocutor-{slug}-{digest}.md"


def daily_filename(day_stamp: str) -> str:
    return f"{day_stamp}.md"


def narrative_path_for_dir(cls, memory_dir: Path, task_id: str | None) -> Path:
    return Path(memory_dir) / "episodic" / cls._narrative_filename(task_id)


def daily_path_for_dir(cls, memory_dir: Path, day_stamp: str) -> Path:
    return Path(memory_dir) / "episodic" / "daily" / cls._daily_filename(day_stamp)


def chat_path_for_dir(cls, memory_dir: Path, chat_id: str) -> Path:
    return Path(memory_dir) / "episodic" / "chat" / cls._chat_filename(chat_id)


def interlocutor_path_for_dir(cls, memory_dir: Path, interlocutor_id: str) -> Path:
    return Path(memory_dir) / "episodic" / "interlocutor" / cls._interlocutor_filename(interlocutor_id)


def legacy_narrative_path_for_dir(cls, memory_dir: Path, task_id: str | None) -> Path:
    return Path(memory_dir) / cls._narrative_filename(task_id)


def task_path(memory, task_id: str | None) -> Path:
    return memory.narrative_path_for_dir(memory._dir, task_id)


def daily_path(memory, day_stamp: str) -> Path:
    return memory.daily_path_for_dir(memory._dir, day_stamp)


def chat_path(memory, chat_id: str) -> Path:
    return memory.chat_path_for_dir(memory._dir, chat_id)


def interlocutor_path(memory, interlocutor_id: str) -> Path:
    return memory.interlocutor_path_for_dir(memory._dir, interlocutor_id)


def legacy_task_path(memory, task_id: str | None) -> Path:
    return memory.legacy_narrative_path_for_dir(memory._dir, task_id)


def resolve_task_path(memory, task_id: str | None) -> Path:
    path = memory._task_path(task_id)
    if path.exists():
        return path
    legacy = memory._legacy_task_path(task_id)
    return legacy if legacy.exists() else path


def iter_legacy_narrative_files(memory) -> list[Path]:
    paths: list[Path] = []
    global_path = memory._legacy_task_path(None)
    if global_path.exists():
        paths.append(global_path)
    paths.extend(sorted(memory._dir.glob("task-*.md")))
    return paths


def iter_narrative_files(memory) -> list[Path]:
    files: dict[str, Path] = {}
    global_path = memory._task_path(None)
    if global_path.exists():
        files[global_path.name] = global_path
    for md_path in sorted(memory._narrative_dir.glob("task-*.md")):
        files.setdefault(md_path.name, md_path)
    for legacy_path in memory._iter_legacy_narrative_files():
        files.setdefault(legacy_path.name, legacy_path)
    return [files[name] for name in sorted(files)]


def migrate_legacy_narrative_files(memory) -> None:
    """将旧版根目录 narrative 文件迁移到 episodic/ 子目录（幂等）。"""
    for legacy_path in memory._iter_legacy_narrative_files():
        target = memory._narrative_dir / legacy_path.name
        if target.exists():
            continue
        try:
            legacy_path.rename(target)
        except OSError as exc:
            _log.warning("[episodic] 迁移 narrative 文件失败: %s -> %s (%s)", legacy_path, target, exc)


def append_markdown_block(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def day_stamp_from_ts(ts: str) -> str:
    return (ts or "").strip()[:10]


def insert_narrative_row(
    memory,
    *,
    task_id: str | None,
    chat_id: str | None,
    interlocutor_id: str | None,
    role: str,
    source_type: str,
    content: str,
    affect_json: str | None,
    ts: str,
) -> int:
    cur = memory._conn.execute(
        "INSERT INTO narrative(task_id, chat_id, interlocutor_id, role, source_type, content, affect, ts)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, chat_id, interlocutor_id, role, source_type, content, affect_json, ts),
    )
    memory._conn.commit()
    return int(cur.lastrowid or 0)


def sync_narrative_fts(
    memory,
    *,
    row_id: int,
    task_id: str | None,
    chat_id: str | None,
    interlocutor_id: str | None,
    role: str,
    content: str,
) -> None:
    memory._conn.execute(
        "INSERT INTO narrative_fts(id, task_id, chat_id, interlocutor_id, role, content) VALUES (?, ?, ?, ?, ?, ?)",
        (row_id, task_id or "", chat_id or "", interlocutor_id or "", role, content),
    )
    memory._conn.commit()


def record(
    memory,
    role: str,
    content: str,
    task_id: str | None = None,
    source_type: str = "",
    affect: dict[str, Any] | None = None,
    *,
    chat_id: str | None = None,
    interlocutor_id: str | None = None,
) -> None:
    """追加一条情节记录（Tulving 1983 四元素绑定）。"""
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    src = source_type or source_from_role(role)

    meta_parts = [f"role={role}", f"src={src}"]
    if chat_id:
        meta_parts.append(f"chat={chat_id}")
    if interlocutor_id:
        meta_parts.append(f"interlocutor={interlocutor_id}")
    if affect:
        v = affect.get("valence")
        a = affect.get("arousal")
        if v is not None and a is not None:
            meta_parts.append(f"affect=({float(v):.2f},{float(a):.2f})")

    meta = " | ".join(meta_parts)
    block = f"\n---\n**[{ts}]** `{meta}`\n\n{content.strip()}\n"

    memory._append_markdown_block(memory._task_path(task_id), block)
    if chat_id and role in {"user", "assistant_reply"}:
        memory._append_markdown_block(memory._chat_path(chat_id), block)
    if interlocutor_id and role in {"user", "assistant_reply"}:
        memory._append_markdown_block(memory._interlocutor_path(interlocutor_id), block)
    memory._append_markdown_block(memory._daily_path(memory._day_stamp_from_ts(ts)), block)

    affect_json = json.dumps(affect, ensure_ascii=False) if affect else None

    with memory._db_session():
        try:
            row_id = memory._insert_narrative_row(
                task_id=task_id,
                chat_id=chat_id,
                interlocutor_id=interlocutor_id,
                role=role,
                source_type=src,
                content=content,
                affect_json=affect_json,
                ts=ts,
            )
        except Exception as narrative_err:
            _log.warning("[episodic] narrative 写入失败（.md 已保留）: %s", narrative_err)
            return

        try:
            memory._sync_narrative_fts(
                row_id=row_id,
                task_id=task_id,
                chat_id=chat_id,
                interlocutor_id=interlocutor_id,
                role=role,
                content=content,
            )
        except Exception as fts_err:
            _log.warning("[episodic] FTS5 写入失败（narrative 已提交，search 将退回 .md 扫描）: %s", fts_err)


def _load_recent_blocks(path: Path, n_recent: int) -> str:
    """从文件尾部逆向分页读取最近 n_recent 条完整情节块（--- 分隔），凑够即停。

    统一机制（Tulving 1983 episode unit）：
    - 每条 --- 块是一个完整的交互事件，是语义原子，不可在块内截断。
    - n_recent 控制注入深度；从尾部分页读取，避免将整个大文件加载进内存。
    - 越新的块越靠近尾部，天然满足 recency bias（Murdock 1962）。
    """
    if not path.exists() or n_recent <= 0:
        return ""
    try:
        file_size = path.stat().st_size
    except OSError:
        return ""
    if file_size == 0:
        return ""

    _CHUNK = 32 * 1024  # 每页 32 KB
    _SEP = b"---"
    blocks: list[str] = []
    tail_buf = b""
    offset = file_size

    with path.open("rb") as fh:
        while offset > 0:
            read_size = min(_CHUNK, offset)
            offset -= read_size
            fh.seek(offset)
            # 当前 chunk 拼上上一轮保留的尾部碎片
            chunk = fh.read(read_size) + tail_buf
            parts = chunk.split(_SEP)
            # parts[0] 可能被 chunk 边界截断（当 offset > 0 时），保留给下轮
            tail_buf = parts[0]
            for raw in reversed(parts[1:]):
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    blocks.append(text)
                if len(blocks) >= n_recent:
                    break
            if len(blocks) >= n_recent:
                break
        # 文件已全部读完时，tail_buf 是第一个块（文件头部）
        if offset == 0 and len(blocks) < n_recent and tail_buf:
            text = tail_buf.decode("utf-8", errors="replace").strip()
            if text:
                blocks.append(text)

    # blocks 从新到旧，逆转为时间升序后拼接
    recent = list(reversed(blocks[:n_recent]))
    return "\n---\n".join(recent)


def load_for_speaker_recognition(memory, interlocutor_id: str | None, *, n_recent: int = 5) -> str:
    """取最近 n_recent 条完整交互块，专用于说话人识别（recognition，非 recall）。

    科学依据：
    - Tulving (1983)：情节记忆的基本单元是事件（episode），按完整块取而非按字节截切。
    - Cowan (2001)：工作记忆有效处理单元约 4 个 chunk；5 条事件是识别任务的实用上限。
    - Liu et al. (2023) "Lost in the Middle"：识别准确率随 context 增长而下降；
      短且聚焦的 context 表现更好。
    """
    if not interlocutor_id:
        return ""
    return _load_recent_blocks(memory._interlocutor_path(interlocutor_id), n_recent)


def load_for_context(memory, task_id: str | None, n_recent: int = 20) -> str:
    """读取情节记忆，注入 LLM context；取最近 n_recent 条完整事件块。"""
    return _load_recent_blocks(memory._resolve_task_path(task_id), n_recent)


def load_for_chat_context(
    memory,
    chat_id: str | None,
    n_recent: int = 20,
    *,
    max_chars: int | None = None,
) -> str:
    """读取 chat 维度的情节连续性，跨 task 保留同一 chat 的完整对话线索。"""
    if not chat_id:
        return ""
    return _load_recent_blocks(memory._chat_path(chat_id), n_recent)


def load_for_interlocutor_context(
    memory,
    interlocutor_id: str | None,
    n_recent: int = 20,
    *,
    max_chars: int | None = None,
) -> str:
    """读取当前交互对象维度的情节连续性，跨 chat 保留同一对象的互动片段。"""
    if not interlocutor_id:
        return ""
    return _load_recent_blocks(memory._interlocutor_path(interlocutor_id), n_recent)


def load_for_task_narrative(memory, task_id: str | None, n_recent: int = 20) -> str:
    """任务叙事模式（Ricoeur 1984）：跨 chat 读取该任务的完整情节流。"""
    return memory.load_for_context(task_id, n_recent)


def load_recent_daily_context(memory, days: int = 2, max_chars: int = 1200) -> str:
    """读取最近若干天的 daily 叙事，用于跨任务的短程连续性。"""
    days = max(1, days)
    if max_chars <= 0:
        max_chars = 999_999_999

    stamps = [
        (datetime.now(UTC) - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(days)
    ]
    sections: list[str] = []
    total_chars = 0

    for stamp in stamps:
        path = memory._daily_path(stamp)
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue

        block = f"[{stamp}]\n{text}"
        if total_chars + len(block) > max_chars:
            break
        sections.append(block)
        total_chars += len(block)
        if total_chars >= max_chars:
            break

    return "\n\n---\n\n".join(sections)


def _daily_search_excerpt(snippet: str, terms: list[str], max_chars: int = _DAILY_SEARCH_EXCERPT_CHARS) -> str:
    text = str(snippet or "").strip()
    if not text or len(text) <= max_chars:
        return text
    lower = text.lower()
    positions = [lower.find(term) for term in terms if term and lower.find(term) >= 0]
    pivot = min(positions) if positions else 0
    half = max_chars // 2
    start = max(0, pivot - half)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def search_recent_daily(memory, query: str, days: int = 2, max_chars: int = 1200) -> str:
    """在最近若干天的 daily 中按 query 检索相关片段。

    用于长期记忆命中不足时的短期补短，避免每轮固定注入整段 recent daily。
    """
    query = (query or "").strip()
    if not query:
        return ""
    days = max(1, days)

    safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
    strict = [t.lower() for t in safe.split() if len(t) >= 2 and not (t.isascii() and len(t) < 5)]
    relaxed = [t.lower() for t in safe.split() if len(t) > 1]
    term_sets = [strict if strict else relaxed]
    if strict and relaxed != strict:
        term_sets.append(relaxed)
    if not term_sets[0]:
        return ""

    scored_hits: list[tuple[int, int, str, list[str]]] = []
    for terms in term_sets:
        scored_hits = []
        for offset in range(days):
            stamp = (datetime.now(UTC) - timedelta(days=offset)).strftime("%Y-%m-%d")
            path = memory._daily_path(stamp)
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for block in reversed(text.split("---")):
                block = block.strip()
                if not block:
                    continue
                body = "\n".join(
                    line for line in block.splitlines()
                    if line.strip() and not line.startswith("**[")
                ).strip()
                if not body:
                    continue
                lower_body = body.lower()
                match_count = sum(1 for term in terms if term in lower_body)
                if match_count <= 0:
                    continue
                snippet = f"[{stamp}]\n{block}"
                scored_hits.append((match_count, -offset, snippet, terms))
        if scored_hits:
            break

    if not scored_hits:
        return ""

    scored_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    hits: list[str] = []
    total = 0
    excerpt_chars = _DAILY_SEARCH_EXCERPT_CHARS if max_chars <= 0 else min(_DAILY_SEARCH_EXCERPT_CHARS, max_chars)
    for _, _, snippet, terms in scored_hits[:_DAILY_SEARCH_MAX_HITS]:
        excerpt = _daily_search_excerpt(snippet, terms, excerpt_chars)
        if max_chars > 0 and total + len(excerpt) > max_chars:
            if hits:
                break
            hits.append(excerpt)
            break
        hits.append(excerpt)
        total += len(excerpt)
        if max_chars > 0 and total >= max_chars:
            return "\n\n---\n\n".join(hits)
    return "\n\n---\n\n".join(hits)


def get_recent_turns(
    memory,
    task_id: str | None = None,
    limit: int = 3,
    *,
    chat_id: str | None = None,
    interlocutor_id: str | None = None,
) -> list[dict[str, Any]]:
    """从 narrative 表返回最近 limit 条对话轮次（用户消息 + 智能体回复）。

    这是 STM 对话缓冲的正确来源——基于情节记忆而非原始 chat_messages 表，
    保留了 Tulving (1983) 四元素绑定中的时间标签和情感状态。

    返回列表按时间升序（最旧→最新），字段:
        role: "user" | "assistant_reply"
        content: str
        ts: str (UTC)
        affect: dict | None  {"valence": float, "arousal": float}
    """
    with memory._db_session():
        try:
            _limit = limit if limit > 0 else 999_999_999
            if chat_id and interlocutor_id:
                sql = (
                    "SELECT role, content, ts, affect FROM narrative "
                    "WHERE chat_id = ? AND interlocutor_id = ? AND role IN ('user', 'assistant_reply') "
                    "ORDER BY id DESC LIMIT ?"
                )
                rows = memory._conn.execute(sql, (chat_id, interlocutor_id, _limit)).fetchall()
            elif chat_id:
                sql = (
                    "SELECT role, content, ts, affect FROM narrative "
                    "WHERE chat_id = ? AND role IN ('user', 'assistant_reply') "
                    "ORDER BY id DESC LIMIT ?"
                )
                rows = memory._conn.execute(sql, (chat_id, _limit)).fetchall()
            elif interlocutor_id:
                sql = (
                    "SELECT role, content, ts, affect FROM narrative "
                    "WHERE interlocutor_id = ? AND role IN ('user', 'assistant_reply') "
                    "ORDER BY id DESC LIMIT ?"
                )
                rows = memory._conn.execute(sql, (interlocutor_id, _limit)).fetchall()
            elif task_id:
                sql = (
                    "SELECT role, content, ts, affect FROM narrative "
                    "WHERE task_id = ? AND role IN ('user', 'assistant_reply') "
                    "ORDER BY id DESC LIMIT ?"
                )
                rows = memory._conn.execute(sql, (task_id, _limit)).fetchall()
            else:
                sql = (
                    "SELECT role, content, ts, affect FROM narrative "
                    "WHERE role IN ('user', 'assistant_reply') "
                    "ORDER BY id DESC LIMIT ?"
                )
                rows = memory._conn.execute(sql, (_limit,)).fetchall()
        except Exception as exc:
            _log.warning("[episodic] get_recent_turns 失败: %s", exc)
            return []

    result: list[dict[str, Any]] = []
    for row in reversed(rows):
        affect: dict[str, Any] | None = None
        if row["affect"]:
            with suppress(Exception):
                affect = json.loads(row["affect"])
        result.append({
            "role": row["role"],
            "content": row["content"] or "",
            "ts": row["ts"] or "",
            "affect": affect,
        })
    return result


def list_tasks(memory) -> list[str]:
    """返回已有情节记忆的任务 ID 列表。"""
    return [p.stem.removeprefix("task-") for p in memory._iter_narrative_files() if p.name.startswith("task-")]


def list_recent_narrative(memory, limit: int = 10) -> list[dict[str, Any]]:
    """返回最新若干条叙事记录（不解释用户时间词，仅供 recent 预热）。"""
    with memory._db_session():
        try:
            rows = memory._conn.execute(
                "SELECT task_id, chat_id, role, content, ts FROM narrative ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []


def query_recent_narrative(memory, hours: int = 24, limit: int = 10) -> list[dict[str, Any]]:
    """时间窗叙事查询：返回最近 hours 小时内的叙事记录（供实体共指消解使用）。"""
    since_dt = datetime.now(UTC) - timedelta(hours=max(1, hours))
    since_str = since_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    with memory._db_session():
        try:
            rows = memory._conn.execute(
                "SELECT task_id, chat_id, role, content, ts FROM narrative"
                " WHERE ts >= ? ORDER BY id DESC LIMIT ?",
                (since_str, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []


def search(
    memory,
    query: str,
    max_chars: int = 2000,
    exclude_task_id: str | None = None,
) -> str:
    """全文检索情节记忆：narrative FTS5（O(log n)）+ .md 文件降级扫描。"""
    if not query.strip():
        return ""
    hits: list[str] = []
    total = 0
    query_stripped = query.strip()

    with memory._db_session():
        safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
        strict = [t for t in safe.split() if len(t) >= 2 and not (t.isascii() and len(t) < 5)]
        terms = strict if strict else [t for t in safe.split() if len(t) > 1]
        if terms:
            fts_query = " OR ".join(terms)
            try:
                rows = memory._conn.execute(
                    "SELECT task_id, chat_id, role, content FROM narrative_fts"
                    " WHERE narrative_fts MATCH ? LIMIT 50",
                    (fts_query,),
                ).fetchall()
                for row in rows:
                    if exclude_task_id and row["task_id"] == exclude_task_id:
                        continue
                    if query_stripped and row["content"].strip() == query_stripped:
                        continue
                    origin = f"task={row['task_id'] or 'global'}"
                    if row["chat_id"]:
                        origin += f" chat={row['chat_id']}"
                    snippet = f"[{origin} role={row['role']}] {row['content']}"
                    hits.append(snippet)
                    total += len(snippet)
                    if total >= max_chars:
                        return "\n\n---\n\n".join(hits)
            except Exception:
                pass

    _MD_SCAN_SIZE_LIMIT = 32 * 1024  # 单文件超过 32KB 跳过降级扫描，避免大文件全量加载
    if total < max_chars:
        keywords = [kw.lower() for kw in query.split() if kw]
        for md_path in memory._iter_narrative_files():
            if exclude_task_id and md_path.name == f"task-{exclude_task_id}.md":
                continue
            try:
                if md_path.stat().st_size > _MD_SCAN_SIZE_LIMIT:
                    continue
                text = md_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for block in text.split("---"):
                block = block.strip()
                if not block:
                    continue
                block_body = "\n".join(
                    line for line in block.splitlines() if line.strip() and not line.startswith("**[")
                ).strip()
                if query_stripped and block_body == query_stripped:
                    continue
                lower = block.lower()
                if all(kw in lower for kw in keywords):
                    snippet = f"[{md_path.name}]\n{block}"
                    hits.append(snippet)
                    total += len(snippet)
                    if total >= max_chars:
                        return "\n\n---\n\n".join(hits)

    return "\n\n---\n\n".join(hits) if hits else ""
