from __future__ import annotations

import json
import logging as _log_sem
import re
import sqlite3
import struct
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from memory.quality_checker import evaluate_retrieval_quality

from . import (
    _EPHEMERAL_MEMORY_KINDS,
    _STABLE_MEMORY_KINDS,
    _STABLE_MEMORY_SOURCES,
    MemoryNode,
    _cosine,
    effective_activation,
)

# 主表查询列（不含 embedding）：embedding 是高维感知数据，只在向量比较时按需加载
_NODE_COLS = "id, kind, title, body, activation, valence, importance, tags, source, created_at"


def _blob_to_vec(raw: bytes | str | list) -> list[float] | None:
    """float32 BLOB → list[float]；兼容旧格式 JSON TEXT 。

    行业惯例：4 bytes/dim BLOB，与 sqlite-vec / pgvector 一致。
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        n = len(b) // 4
        if n == 0:
            return None
        return list(struct.unpack(f"{n}f", b[:n * 4]))
    if isinstance(raw, str):
        try:
            return json.loads(raw)  # 当心旧数据 JSON TEXT
        except Exception:
            return None
    if isinstance(raw, list):
        return raw

    return None

_log = _log_sem.getLogger("lingzhou.memory.semantic")

# 向量扫描预筛前置层：每批从 DB 拉多少行 BLOB 做余弦计算
_VEC_SCAN_PAGE = 500
_FTS5_OPERATOR_TOKENS = frozenset({"and", "or", "not", "near"})


def _quote_fts5_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _vec_scan_candidates(
    self,
    query_vec: list[float],
    *,
    modality: str = "text",
    source: str | None = None,
    top_n: int = 200,
) -> list[str]:
    """node_embeddings 表分批全量余弦扫描，返回最相似 top_n 个 node_id。

    内存安全：每轮最多 _VEC_SCAN_PAGE 条 BLOB 在 Python 中，计算完即丢弃。
    100万节点 × 6KB BLOB 该用 disk streaming，现阶段规模 ~10k 内存安全。
    """
    q_dim = len(query_vec)
    scores: list[tuple[float, str]] = []  # (cosine_sim, node_id)
    offset = 0
    src_clause = " AND n.source = ?" if source else ""
    sql = (
        f"SELECT e.node_id, e.vector"
        f" FROM node_embeddings e"
        f" JOIN nodes n ON n.id = e.node_id"
        f" WHERE e.modality = ?{src_clause}"
        f" LIMIT ? OFFSET ?"
    )
    while True:
        params: list[Any] = [modality]
        if source:
            params.append(source)
        params += [_VEC_SCAN_PAGE, offset]
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except Exception:
            break
        if not rows:
            break
        for row in rows:
            nid, vec_raw = row[0], row[1]
            if not vec_raw:
                continue
            vec = _blob_to_vec(vec_raw)
            if vec is None or len(vec) != q_dim:
                continue
            sim = _cosine(query_vec, vec)
            scores.append((sim, nid))
        if len(rows) < _VEC_SCAN_PAGE:
            break
        offset += _VEC_SCAN_PAGE
    scores.sort(key=lambda x: x[0], reverse=True)
    return [nid for _, nid in scores[:top_n]]


def _fallback_candidates(
    self,
    query_vec: list[float] | None,
    *,
    kind: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    task_id: str | int | None = None,
    id_prefix: str | None = None,
    path_prefix: str | None = None,
    top_n: int = 200,
) -> list[MemoryNode]:
    """公共兜底路径：
    1. 有 query_vec → 先向量扫描预筛 top_n IDs，再加载完整节点 + 条件过滤。
    2. 无 query_vec → SQL 层 activation 排序 + 条件过滤，带 embedding。
    内存安全：向量扫描分批处理，_load_filtered 有字段限制。
    """
    if query_vec is not None:
        # 有向量：先扫描所有已嵌入节点的余弦相似度，取 top_n IDs
        candidate_ids = self._vec_scan_candidates(query_vec, source=source, top_n=top_n)
        if candidate_ids:
            nodes = self._load_by_ids(candidate_ids)
            # 向量扫描不含其他字段过滤，在 Python 层补充
            if any((kind, tag, task_id, id_prefix, path_prefix)):
                nodes = [
                    n for n in nodes
                    if self._matches_filters(
                        n, kind=kind, tag=tag, source=source,
                        task_id=task_id, id_prefix=id_prefix, path_prefix=path_prefix,
                    )
                ]
            if nodes:
                return nodes
        # 向量扫描无结果（无 embedding 数据）→ 降级到 activation 排序
    # 无向量（或向量扫描失败）：纲 activation 全局排序，带 embedding 附加
    _fb = self._load_filtered(
        kind=kind, tag=tag, source=source, task_id=task_id,
        id_prefix=id_prefix, path_prefix=path_prefix, limit=top_n,
    )
    return self._load_by_ids([n.id for n in _fb]) if _fb else []


def _node_has_rankable_embedding(node: MemoryNode, query_modality: str = "text") -> bool:
    emb_dict: dict[tuple[str, str], list[float]] = getattr(node, "embeddings", {}) or {}
    if any(mod == query_modality and vec for (mod, _), vec in emb_dict.items()):
        return True
    legacy_raw = getattr(node, "embedding", None)
    return _blob_to_vec(legacy_raw) is not None if legacy_raw is not None else False


def retrieve(
    self,
    query: str,
    top_k: int = 5,
    *,
    kind: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    task_id: str | int | None = None,
    path_prefix: str | None = None,
    id_prefix: str | None = None,
) -> list[dict[str, Any]]:
    with self._db_session():
        query_vec: list[float] | None = None
        query_vec_loaded = False

        def _load_query_vec() -> list[float] | None:
            nonlocal query_vec, query_vec_loaded
            if query_vec_loaded:
                return query_vec
            query_vec_loaded = True
            if self._embed_fn is not None:
                with suppress(Exception):
                    query_vec = self._embed_fn(query)
            return query_vec

        has_filters = any((kind, tag, source, task_id, path_prefix, id_prefix))
        candidate_ids = self._fts_candidates(query, limit=100 if has_filters else 50)
        if candidate_ids:
            nodes = self._load_by_ids(candidate_ids)
            if has_filters:
                nodes = [
                    n for n in nodes
                    if self._matches_filters(
                        n, kind=kind, tag=tag, source=source,
                        task_id=task_id, path_prefix=path_prefix, id_prefix=id_prefix,
                    )
                ]
            # FTS5 命中但被过滤器全部过滤掉 → SQL 层按条件兜底，不全表扫描
            if not nodes:
                nodes = self._fallback_candidates(
                    None, kind=kind, tag=tag, source=source,
                    task_id=task_id, id_prefix=id_prefix, path_prefix=path_prefix,
                )
        else:
            # FTS5 无命中 → 向量预筛 + 条件过滤兜底
            nodes = self._fallback_candidates(
                None, kind=kind, tag=tag, source=source,
                task_id=task_id, id_prefix=id_prefix, path_prefix=path_prefix,
            )
        if not nodes:
            return []
        if self._embed_fn is not None and any(_node_has_rankable_embedding(node) for node in nodes):
            _load_query_vec()
        scored = [(self._score(query, n, query_vec=query_vec), n) for n in nodes]
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = []
        for score, node in scored[:top_k]:
            item = node.to_dict()
            item["score"] = round(float(score), 4)
            retrieved.append(item)

        if _log.isEnabledFor(_log_sem.DEBUG):
            qm = evaluate_retrieval_quality(query, retrieved, self._decay_lambda)
            _log.debug("[semantic.retrieve] quality=%s", qm.get("overall_score", 0))
        return retrieved


def _matches_filters(
    node: MemoryNode,
    *,
    kind: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    task_id: str | int | None = None,
    path_prefix: str | None = None,
    id_prefix: str | None = None,
) -> bool:
    if kind and node.kind != str(kind).strip():
        return False
    if tag:
        normalized_tag = str(tag).strip()
        if normalized_tag and normalized_tag not in node.tags:
            return False
    if source:
        normalized_source = str(source).strip()
        node_source = str(getattr(node, "source", "")).strip()
        if normalized_source and normalized_source != node_source:
            return False
    if task_id is not None:
        expected_task_tag = f"task:{str(task_id).strip()}"
        if expected_task_tag not in node.tags:
            return False
    if id_prefix:
        normalized_id_prefix = str(id_prefix).strip()
        if normalized_id_prefix and not node.id.startswith(normalized_id_prefix):
            return False
    if path_prefix:
        normalized_path = str(path_prefix).strip().replace("\\", "/")
        if normalized_path:
            haystack = [
                node.title.replace("\\", "/"),
                node.body.replace("\\", "/"),
                *(tag_item.replace("\\", "/") for tag_item in node.tags),
            ]
            if not any(normalized_path in item for item in haystack):
                return False
    return True


def retrieve_multi_anchor(
    self, anchors: list[str], top_k: int = 5, convergence_bonus: float = 0.15, source: str | None = None
) -> list[dict[str, Any]]:
    with self._db_session():
        start_t0 = time.perf_counter()
        valid_anchors = [a for a in anchors if a and a.strip()]
        if not valid_anchors:
            return []
        _log.info(
            "[semantic.multi_anchor] start anchors=%d top_k=%d source=%s",
            len(valid_anchors),
            top_k,
            str(source or ""),
        )
        all_ids: list[str] = []
        seen: set[str] = set()
        retrieval_route = "fts"
        for anchor in valid_anchors:
            for nid in self._fts_candidates(anchor, limit=30):
                if nid not in seen:
                    seen.add(nid)
                    all_ids.append(nid)
        if all_ids:
            nodes = self._load_by_ids(all_ids)
            if source:
                nodes = [n for n in nodes if getattr(n, "source", "") == source]
        else:
            # FTS5 对所有 anchor 均无命中：先按 activation 兜底，后按需触发向量评分。
            # 仅在候选节点具备 rankable embedding 时才触发 embed 计算。
            valid_ids: set[str] = set()
            retrieval_route = "vec_scan"
            _fb = self._load_filtered(source=source, limit=200)
            nodes = self._load_by_ids([n.id for n in _fb]) if _fb else []

            use_vector_scoring = self._embed_fn is not None and nodes and any(
                _node_has_rankable_embedding(n) for n in nodes
            )
            if use_vector_scoring:
                retrieval_route = "vec_scan"
                for anchor in valid_anchors:
                    with suppress(Exception):
                        anchor_vec = self._embed_fn(anchor)
                        seen_ids = set()
                        for nid in self._vec_scan_candidates(anchor_vec, source=source, top_n=100):
                            seen_ids.add(nid)
                        if seen_ids:
                            # 与 FTS 命中路径保持接口一致：保留并集结果
                            # 并允许后续按 source 二次过滤。
                            valid_ids |= seen_ids
            else:
                retrieval_route = "filtered_fallback"

            if use_vector_scoring and valid_ids:
                nodes = self._load_by_ids(list(valid_ids))
                if source:
                    nodes = [n for n in nodes if getattr(n, "source", "") == source]
        if not nodes:
            _log.info(
                "[semantic.multi_anchor] done dt=%.3fs route=%s nodes=0 hits=0",
                time.perf_counter() - start_t0,
                retrieval_route,
            )
            return []

        use_vector_scoring = self._embed_fn is not None and any(_node_has_rankable_embedding(node) for node in nodes)
        anchor_vecs: dict[str, list[float] | None] = {}
        if use_vector_scoring:
            for anchor in valid_anchors:
                with suppress(Exception):
                    anchor_vecs[anchor] = self._embed_fn(anchor)

        best_score: dict[str, float] = {}
        hit_count: dict[str, int] = {}
        for anchor in valid_anchors:
            query_vec = anchor_vecs.get(anchor) if use_vector_scoring else None
            for node in nodes:
                s = self._score(anchor, node, query_vec=query_vec)
                if s > 0:
                    if node.id not in best_score or s > best_score[node.id]:
                        best_score[node.id] = s
                    hit_count[node.id] = hit_count.get(node.id, 0) + 1

        if not best_score:
            return []

        node_map = {n.id: n for n in nodes}
        final: list[tuple[float, MemoryNode]] = []
        for nid, base in best_score.items():
            hits = hit_count.get(nid, 1)
            score = base * (1.0 + convergence_bonus * (hits - 1))
            final.append((score, node_map[nid]))

        final.sort(key=lambda x: x[0], reverse=True)
        retrieved = []
        for score, node in final[:top_k]:
            item = node.to_dict()
            item["score"] = round(float(score), 4)
            retrieved.append(item)

        _log.info(
            "[semantic.multi_anchor] done dt=%.3fs route=%s nodes=%d hits=%d vector_scoring=%s",
            time.perf_counter() - start_t0,
            retrieval_route,
            len(nodes),
            len(retrieved),
            use_vector_scoring,
        )

        if _log.isEnabledFor(_log_sem.DEBUG):
            combined_query = " ".join(valid_anchors)
            qm = evaluate_retrieval_quality(combined_query, retrieved, self._decay_lambda)
            _log.debug("[semantic.multi_anchor] quality=%s", qm.get("overall_score", 0))
        return retrieved


def store_reflection(self, kind: str, insight: str, valence: float = 0.5) -> str:
    node_id = f"reflection-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    self.upsert(MemoryNode(
        id=node_id,
        kind="learned_insight",
        title=f"[{kind}] [{node_id[-6:]}]",
        body=insight.strip(),
        activation=0.8,
        valence=valence,
        tags=[kind],
    ))
    return node_id


def list_reflections(self, limit: int = 10) -> list[MemoryNode]:
    with self._db_session():
        try:
            rows = self._conn.execute(
                f"SELECT {_NODE_COLS} FROM nodes "
                "WHERE kind = 'learned_insight' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]
        except Exception:
            return []


# unicode61 将 CJK Unified Ideographs 每个字符单独分词
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def _fts_candidates(self, query: str, limit: int = 50) -> list[str]:
    if not self._fts5_ok:
        # 感知自愈：冷却期过后重新探测 FTS5 可用性，而非永久失聪
        if time.time() < self._fts5_retry_after:
            return []
        try:
            self._conn.execute("SELECT * FROM nodes_fts LIMIT 0")
            self._fts5_ok = True
        except Exception:
            self._fts5_retry_after = time.time() + 300
            return []
    safe = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
    # unicode61 对 CJK 逐字索引，查询也须逐字拆开，否则整词 MATCH 必然空返回
    raw_tokens: list[str] = []
    for word in safe.split():
        if _CJK_RE.search(word):
            raw_tokens.extend(c for c in word if c.strip())
        else:
            raw_tokens.append(word)
    # 过滤：保留 CJK 单字 + 长度 >= 2 的非 ASCII 词 + 长度 >= 2 的英文词
    terms = [
        t for t in raw_tokens
        if t.strip() and t.lower() not in _FTS5_OPERATOR_TOKENS and (not t.isascii() or len(t) >= 2)
    ]
    if not terms:
        return []
    fts_query = " OR ".join(_quote_fts5_term(t) for t in terms)
    try:
        rows = self._conn.execute(
            "SELECT id FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [r[0] for r in rows]
    except Exception as exc:
        self._fts5_ok = False
        self._fts5_retry_after = time.time() + 300
        _log.warning("[semantic] FTS5 感知受损，5分钟后自愈重试: %s", exc)
        return []


def _load_by_ids(self, ids: list[str]) -> list[MemoryNode]:
    if not ids:
        return []
    # 主字段不含 embedding，避免把旧 TEXT 列的高维 JSON 无条件装入 Python
    try:
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE id IN ({placeholders})", ids
        ).fetchall()
        nodes = [self._row_to_node(r) for r in rows]
    except Exception:
        return []
    if self._embed_fn is not None and nodes:
        # 从多模态 embedding 表按需附加；兼容旧 nodes.embedding 回退
        try:
            emb_rows = self._conn.execute(
                f"SELECT node_id, modality, model, vector"
                f" FROM node_embeddings WHERE node_id IN ({placeholders})",
                ids,
            ).fetchall()
            embeddings_by_node: dict[str, dict[tuple[str, str], list[float]]] = {}
            for er in emb_rows:
                nid, modality, model, vec_raw = er[0], er[1], er[2], er[3]
                if not vec_raw:
                    continue
                vec = _blob_to_vec(vec_raw)  # BLOB → list[float]
                if vec is not None:
                    embeddings_by_node.setdefault(nid, {})[(modality, model)] = vec
            if embeddings_by_node:
                for node in nodes:
                    if node.id in embeddings_by_node:
                        node.__dict__["embeddings"] = embeddings_by_node[node.id]
            else:
                # node_embeddings 表尚无数据（旧实例）→ 回退读 nodes.embedding 列
                legacy_rows = self._conn.execute(
                    f"SELECT id, embedding FROM nodes WHERE id IN ({placeholders})"
                    " AND embedding IS NOT NULL",
                    ids,
                ).fetchall()
                for lr in legacy_rows:
                    nid, emb_raw = lr[0], lr[1]
                    if not emb_raw:
                        continue
                    vec = _blob_to_vec(emb_raw)  # 旧格式可能是 JSON TEXT 或 BLOB
                    if vec is not None:
                        for node in nodes:
                            if node.id == nid:
                                node.__dict__.setdefault(
                                    "embeddings", {}
                                )[("text", "legacy")] = vec
                                break
        except Exception:
            pass
    return nodes


def _load_filtered(
    self,
    *,
    kind: str | None = None,
    tag: str | None = None,
    source: str | None = None,
    task_id: str | int | None = None,
    id_prefix: str | None = None,
    path_prefix: str | None = None,
    limit: int = 200,
) -> list[MemoryNode]:
    """SQL 层过滤 + activation 排序，只取 limit 条进 Python。

    kind / source / id_prefix / tag / task_id 可完全下推到 SQL WHERE。
    path_prefix 搜索 title/body/tags 文本内容，在 Python 层补充过滤。
    """
    where: list[str] = []
    params: list[Any] = []
    if kind:
        where.append("kind = ?")
        params.append(str(kind).strip())
    if source:
        where.append("source = ?")
        params.append(str(source).strip())
    if id_prefix:
        where.append("id LIKE ?")
        params.append(str(id_prefix).strip() + "%")
    if task_id is not None:
        # tags 存为 JSON 数组，如 '["task:42","wm_promoted"]'
        where.append('tags LIKE ?')
        params.append(f'%"task:{str(task_id).strip()}"%')
    if tag:
        where.append('tags LIKE ?')
        params.append(f'%"{str(tag).strip()}"%')
    sql = f"SELECT {_NODE_COLS} FROM nodes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY activation DESC LIMIT ?"
    params.append(limit)
    try:
        rows = self._conn.execute(sql, params).fetchall()
        nodes = [self._row_to_node(r) for r in rows]
    except Exception:
        return []
    # path_prefix 需扫 title/body/tags 文本，只对已缩小的集合做 Python 过滤
    if path_prefix:
        nodes = [n for n in nodes if self._matches_filters(n, path_prefix=path_prefix)]
    return nodes


def _row_to_node(row: sqlite3.Row) -> MemoryNode:
    d: dict[str, Any] = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    node = MemoryNode.from_dict(d)
    emb = d.get("embedding")
    if emb is not None:
        node.__dict__["embedding"] = emb
    return node


def _score(
    self,
    query: str,
    node: MemoryNode,
    query_vec: list[float] | None = None,
    query_modality: str = "text",
) -> float:
    eff_act = effective_activation(node, self._decay_lambda)
    q_tokens = set(re.findall(r"\w+", query.lower()))
    n_tokens = set(re.findall(r"\w+", (node.title + " " + node.body).lower()))
    if not q_tokens or not n_tokens:
        kw_score = 0.1
    else:
        kw_score = len(q_tokens & n_tokens) / len(q_tokens | n_tokens)
    source_score = self._source_score(node)
    temporal_score = self._temporal_score(node)
    text_score = max(0.0, kw_score * 0.55 + eff_act * 0.25 + source_score + temporal_score)

    if query_vec is None:
        return text_score

    # 从多模态 embeddings dict 选取匹配的向量
    # 优先: (modality, '') 精确匹配，其次: (modality, 'legacy')，最后: 同 modality 任意 model
    node_vec: list[float] | None = None
    emb_dict: dict[tuple[str, str], list[float]] = getattr(node, "embeddings", {})
    if emb_dict:
        node_vec = (
            emb_dict.get((query_modality, ""))
            or emb_dict.get((query_modality, "legacy"))
        )
        if node_vec is None:
            # 同模态任意 model
            for (mod, _), v in emb_dict.items():
                if mod == query_modality:
                    node_vec = v
                    break
    # 回退：旧 legacy embedding 字段（迁移前可能存在）
    if node_vec is None:
        legacy_raw = getattr(node, "embedding", None)
        if legacy_raw is not None:
            node_vec = _blob_to_vec(legacy_raw)  # 兼容 JSON TEXT 和 BLOB

    if node_vec is None or len(node_vec) != len(query_vec):
        return text_score
    try:
        cos_sim = _cosine(query_vec, node_vec)
        w = self._embedding_weight
        return (1 - w) * text_score + w * cos_sim
    except Exception:
        return text_score


def _source_score(self, node: MemoryNode) -> float:
    score = 0.0
    stable_kind = node.kind in _STABLE_MEMORY_KINDS
    if node.kind in _STABLE_MEMORY_KINDS:
        score += self._source_weight
    elif node.kind in _EPHEMERAL_MEMORY_KINDS:
        score -= self._source_weight * 0.4

    node_source = str(getattr(node, "source", "") or "").strip()
    if node_source in _STABLE_MEMORY_SOURCES:
        score += self._source_weight * 0.5
        if stable_kind:
            score += self._source_weight * 0.25

    tags = set(getattr(node, "tags", []) or [])
    if "wm_promoted" in tags and node.kind not in _EPHEMERAL_MEMORY_KINDS:
        score += self._source_weight * 0.25
    return score


def _temporal_score(self, node: MemoryNode) -> float:
    if self._temporal_weight <= 0:
        return 0.0
    age_days = self._node_age_days(node)
    if age_days is None:
        return 0.0

    normalized = min(age_days / self._temporal_window_days, 1.0)
    freshness = max(0.0, 1.0 - normalized)
    if node.kind in _STABLE_MEMORY_KINDS:
        return self._temporal_weight * normalized
    if node.kind in _EPHEMERAL_MEMORY_KINDS:
        return self._temporal_weight * 0.35 * freshness
    return self._temporal_weight * 0.15 * freshness


def _node_age_days(node: MemoryNode) -> float | None:
    try:
        created = datetime.fromisoformat(node.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400)
    except Exception:
        return None


def get_unembedded(
    self, modality: str = "text", model: str = "", limit: int = 20
) -> list[tuple[str, str]]:
    """返回尚无指定 (modality, model) embedding 的节点 (id, text) 列表。"""
    with self._db_session():
        try:
            rows = self._conn.execute(
                "SELECT n.id, n.title, n.body FROM nodes n"
                " WHERE NOT EXISTS ("
                "   SELECT 1 FROM node_embeddings e"
                "   WHERE e.node_id = n.id AND e.modality = ? AND e.model = ?"
                " ) LIMIT ?",
                (modality, model, limit),
            ).fetchall()
            return [(r[0], (r[1] or "") + " " + (r[2] or "")) for r in rows]
        except Exception:
            return []


def set_embedding(
    self, node_id: str, vec: list[float],
    modality: str = "text", model: str = "",
) -> None:
    """写入指定 (modality, model) 的 embedding 到 node_embeddings 表。
    vector 存 float32 BLOB，与 sqlite-vec / pgvector 行业惯例一致。
    """
    with self._db_session():
        try:
            blob = struct.pack(f"{len(vec)}f", *vec)
            self._conn.execute(
                "INSERT OR REPLACE INTO node_embeddings"
                " (node_id, modality, model, dim, vector, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, modality, model, len(vec), blob,
                 datetime.now(UTC).isoformat()),
            )
            self._conn.commit()
            # 同步更新 JSON 文件（向后兼容：保留 embedding 字段供索引重建使用）
            if modality == "text":
                json_path = self._dir / f"{node_id}.json"
                if json_path.exists():
                    try:
                        d = json.loads(json_path.read_text(encoding="utf-8"))
                        d["embedding"] = vec
                        json_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
        except Exception:
            pass
