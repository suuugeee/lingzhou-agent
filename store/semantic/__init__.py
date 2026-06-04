"""store.semantic — 语义记忆（SemanticMemory）。

双层存储设计：
  1. nodes/{id}.json  — 运行期语义节点源数据（首先写入，可重建）
  2. semantic.db      — 搜索索引层（由 json 派生，可完全重建，删除后无数据丢失）

恼复路径： semantic.db 损坏 → 启动时自动检测 → 删除并重建 → 从 nodes/*.json 重导入。
检索：FTS5 关键词评分（标准库，零依赖），接口稳定，内部可替换。

说明：这里的 nodes/ 位于配置的 memory_dir 下（默认 ~/.lingzhou/memory/nodes），
不是 workspace_dir，也不应该作为源码仓库的一部分提交。
"""
from __future__ import annotations

import logging as _log_sem
import math
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_log = _log_sem.getLogger("lingzhou.memory.semantic")

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    activation  REAL NOT NULL DEFAULT 0.5,
    valence     REAL NOT NULL DEFAULT 0.5,
    importance  REAL NOT NULL DEFAULT 0.0,
    tags        TEXT NOT NULL DEFAULT '[]',
    source      TEXT NOT NULL DEFAULT '',
    embedding   TEXT,
    created_at  TEXT NOT NULL
);
"""


def _parse_table_cols(ddl: str) -> dict[str, str]:
    """从 CREATE TABLE DDL 中提取列名 → 定义的映射，用于 schema reconciler。"""
    m = re.search(r"CREATE TABLE[^(]*\((.*?)\);", ddl, re.DOTALL | re.IGNORECASE)
    if not m:
        return {}
    cols: dict[str, str] = {}
    for line in m.group(1).split(","):
        line = line.strip()
        if not line:
            continue
        if re.match(r"(PRIMARY\s+KEY|UNIQUE|CHECK|FOREIGN\s+KEY)\b", line, re.IGNORECASE):
            continue
        col_m = re.match(r"^(\w+)\s+(.*)", line)
        if col_m:
            cols[col_m.group(1)] = col_m.group(2).strip()
    return cols


_DDL_FTS5 = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED,
    title,
    body,
    tags,
    tokenize='unicode61'
);
"""

# 多模态 embedding 表：PRIMARY KEY (node_id, modality, model)
# modality: 'text' | 'image' | 'audio' | 'video'
# model:    e.g. 'openai/text-embedding-3-small', 'openai/clip-vit-base-patch32', ''
# vector 存 float32 BLOB（4 bytes/dim），与 sqlite-vec / pgvector 行业惯例一致：
#   - 1536 dim → 6KB BLOB vs ~15KB JSON TEXT
#   - struct.unpack 反序列化比 json.loads 快 ~10x
#   - BLOB 长度 ÷ 4 == dim，天然维度校验，无 JSON 精度损失
# nodes.embedding 列保留（回滚安全窗口），新写入只走此表。
_DDL_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id    TEXT NOT NULL,
    modality   TEXT NOT NULL DEFAULT 'text',
    model      TEXT NOT NULL DEFAULT '',
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (node_id, modality, model)
);
CREATE INDEX IF NOT EXISTS idx_node_emb_modality ON node_embeddings(node_id, modality);
"""

_STABLE_MEMORY_KINDS = frozenset({
    "fact",
    "interlocutor",
    "person",
    "daily_summary",
    "task_summary",
    "learned_insight",
    "self_model_signal",
    "consolidated_insight",
    "control_rule",
    "learned_skill",
})

_EPHEMERAL_MEMORY_KINDS = frozenset({
    "event",
    "task_progress",
    "run_result",
    "sensor_snapshot",
    "delegated_result",
})

_STABLE_MEMORY_SOURCES = frozenset({
    "wm_consolidation",
    "daily_consolidation",
    "memory.add_semantic",
    "manual",
    "reflection",
})


@dataclass
class MemoryNode:
    id: str
    kind: str
    title: str
    body: str
    activation: float = 0.5
    valence: float = 0.5
    importance: float = 0.0
    tags: list[str] = field(default_factory=list[str])
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = asdict(self)  # type: ignore[return-value]
        body = d.get("body") or ""
        d["body_preview"] = body[:300].replace("\n", " ").strip() if len(body) > 300 else body
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def effective_activation(node: MemoryNode, decay_lambda: float) -> float:
    """分级衰减：高重要性节点衰减更慢，并享有激活度保护下限。"""
    importance = max(0.0, min(1.0, float(getattr(node, "importance", 0.0) or 0.0)))
    if decay_lambda <= 0:
        return node.activation
    try:
        created = datetime.fromisoformat(node.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        days = max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400)
        effective_lambda = decay_lambda * (1.0 - importance)
        raw = node.activation * math.exp(-effective_lambda * days)
        if importance >= 0.9:
            raw = max(raw, 0.5)
        elif importance >= 0.7:
            raw = max(raw, 0.3)
        return min(raw, node.activation)
    except Exception:
        return node.activation


_effective_activation = effective_activation


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticMemory:
    def __init__(
        self,
        memory_dir: Path,
        decay_lambda: float = 0.1,
        db_path: Path | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        embedding_weight: float = 0.3,
        source_weight: float = 0.12,
        temporal_weight: float = 0.08,
        temporal_window_days: float = 7.0,
        startup_maintenance_seconds: float = 2.0,
    ) -> None:
        self._dir = memory_dir / "nodes"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._decay_lambda = decay_lambda
        self._db_path = db_path or (memory_dir / "semantic.db")
        self._fts5_ok: bool = False
        self._fts5_retry_after: float = 0.0  # FTS5 感知受损后的自愈冷却截止时间
        if embed_fn is not None:
            _log.info("[semantic] 向量混合检索已启用（实验性，embedding_weight=%.2f）", embedding_weight)
        self._embed_fn = embed_fn
        self._embedding_weight = embedding_weight
        self._source_weight = max(0.0, float(source_weight))
        self._temporal_weight = max(0.0, float(temporal_weight))
        self._temporal_window_days = max(0.1, float(temporal_window_days))
        self._db_lock = threading.RLock()
        self._conn = None
        self._session_depth = 0
        self._maintenance_deferred = False
        self._maintenance_thread: threading.Thread | None = None
        init_started = time.monotonic()
        with self._db_session():
            stage_started = time.monotonic()
            self._migrate()
            _log.info("[semantic] 启动阶段 migrate 完成 dt=%.3fs", time.monotonic() - stage_started)
            stage_started = time.monotonic()
            self._sync_from_files(max_seconds=startup_maintenance_seconds)
            _log.info("[semantic] 启动阶段 sync_from_files 完成 dt=%.3fs", time.monotonic() - stage_started)
            stage_started = time.monotonic()
            self._migrate_interlocutor_profiles(max_seconds=startup_maintenance_seconds)
            _log.info("[semantic] 启动阶段 migrate_profiles 完成 dt=%.3fs", time.monotonic() - stage_started)
            stage_started = time.monotonic()
            self._validate_and_repair_index()
            _log.info("[semantic] 启动阶段 validate_index 完成 dt=%.3fs", time.monotonic() - stage_started)
        _log.info("[semantic] 启动完成 dt=%.3fs", time.monotonic() - init_started)
        if self._maintenance_deferred:
            self._start_deferred_maintenance()

    def _start_deferred_maintenance(self) -> None:
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            return

        def _worker() -> None:
            try:
                self._run_deferred_maintenance()
            except Exception:
                _log.exception("[semantic] 后台索引恢复失败")

        self._maintenance_thread = threading.Thread(
            target=_worker,
            name="lingzhou-semantic-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()

    # These placeholders are replaced by bind_semantic_memory() at import time.
    # Keep them to satisfy static analysis for runtime-bound methods used in __init__.
    @contextmanager
    def _db_session(self):
        raise RuntimeError("semantic db session binding missing")
        yield None

    def _migrate(self) -> None:
        raise RuntimeError("semantic migrate binding missing")

    def _sync_from_files(self, max_seconds: float | None = None) -> None:
        raise RuntimeError("semantic sync binding missing")

    def _migrate_interlocutor_profiles(self, max_seconds: float | None = None) -> None:
        raise RuntimeError("semantic interlocutor migration binding missing")

    def _validate_and_repair_index(self) -> None:
        raise RuntimeError("semantic index validation binding missing")

    def _run_deferred_maintenance(self) -> None:
        raise RuntimeError("semantic deferred maintenance binding missing")


def _bind_semantic_memory() -> None:
    """延迟绑定：在模块构建完成后再把 db/query impl 注入到 SemanticMemory。"""
    from .impl import bind_semantic_memory

    bind_semantic_memory(SemanticMemory)


_bind_semantic_memory()

__all__ = ["MemoryNode", "SemanticMemory", "effective_activation"]
