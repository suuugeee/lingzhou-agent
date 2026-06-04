from __future__ import annotations

from . import SemanticMemory, db, query


def bind_semantic_memory(cls: type[SemanticMemory]) -> None:
    cls._conn = property(db._conn_getter, db._conn_setter)

    cls._normalize_interlocutor_tags = staticmethod(db._normalize_interlocutor_tags)
    cls._is_legacy_interlocutor_profile = classmethod(db._is_legacy_interlocutor_profile)
    cls._matches_filters = staticmethod(query._matches_filters)
    cls._row_to_node = staticmethod(query._row_to_node)
    cls._node_age_days = staticmethod(query._node_age_days)

    cls._migrate_interlocutor_profiles = db._migrate_interlocutor_profiles
    cls._db_session = db._db_session
    cls.close = db.close

    cls._open_db = db._open_db
    cls._migrate = db._migrate
    cls._setup_fts5 = db._setup_fts5
    cls._setup_embeddings_table = db._setup_embeddings_table
    cls._migrate_embeddings = db._migrate_embeddings
    cls._connect = db._connect
    cls._sync_from_files = db._sync_from_files
    cls._validate_and_repair_index = db._validate_and_repair_index
    cls._run_deferred_maintenance = db._run_deferred_maintenance
    cls.rebuild_index = db.rebuild_index
    cls._db_upsert = db._db_upsert
    cls._sync_node_fts = db._sync_node_fts

    cls.fts5_ok = property(db.fts5_ok)
    cls.decay_lambda = property(db.decay_lambda)
    cls.stats = db.stats
    cls.upsert = db.upsert
    cls.get = db.get
    cls.find_by_title = db.find_by_title

    cls.retrieve = query.retrieve
    cls.retrieve_multi_anchor = query.retrieve_multi_anchor
    cls.store_reflection = query.store_reflection
    cls.list_reflections = query.list_reflections

    cls._fts_candidates = query._fts_candidates
    cls._vec_scan_candidates = query._vec_scan_candidates
    cls._fallback_candidates = query._fallback_candidates
    cls._load_by_ids = query._load_by_ids
    cls._load_filtered = query._load_filtered
    cls._score = query._score
    cls._source_score = query._source_score
    cls._temporal_score = query._temporal_score

    cls.get_unembedded = query.get_unembedded
    cls.set_embedding = query.set_embedding
