from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path


def test_compact_runtime_db_dry_run_and_apply():
    from core.maintenance import compact_runtime_db

    huge = "A" * 20000 + "TAIL"
    items = [{"index": idx, "value": huge} for idx in range(90)]
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "runtime.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}')")
            conn.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}')")
            conn.execute("CREATE TABLE facts (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '', scope TEXT, updated_at TEXT)")
            conn.execute("CREATE TABLE failures (id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}')")
            conn.execute("CREATE TABLE life_ledger (id INTEGER PRIMARY KEY, value TEXT NOT NULL DEFAULT '')")
            conn.execute(
                "INSERT INTO runs (id, data) VALUES (?, ?)",
                (1, json.dumps({"output_json": {"summary": huge, "items": items}}, ensure_ascii=False)),
            )
            conn.execute(
                "INSERT INTO tasks (id, data) VALUES (?, ?)",
                (1, json.dumps({"result_json": {"summary": huge}}, ensure_ascii=False)),
            )
            conn.execute(
                "INSERT INTO facts (key, value, scope, updated_at) VALUES (?, ?, 'system', '')",
                ("durable_failure:demo", json.dumps({"last_summary": huge, "count": 3}, ensure_ascii=False)),
            )
            conn.execute(
                "INSERT INTO failures (id, data) VALUES (?, ?)",
                (1, json.dumps({"summary": huge, "context": huge}, ensure_ascii=False)),
            )
            conn.execute("INSERT INTO life_ledger (id, value) VALUES (?, ?)", (1, huge))
            conn.commit()
        finally:
            conn.close()

        dry = compact_runtime_db(db_path)
        assert dry["dry_run"] is True
        assert dry["changed_rows"] == 5
        assert dry["saved_bytes"] > 0

        conn = sqlite3.connect(db_path)
        try:
            raw_fact = conn.execute("SELECT value FROM facts WHERE key='durable_failure:demo'").fetchone()[0]
            assert "last_summary" in raw_fact
        finally:
            conn.close()

        applied = compact_runtime_db(db_path, apply=True)
        assert applied["dry_run"] is False
        assert applied["changed_rows"] == 5

        conn = sqlite3.connect(db_path)
        try:
            run_data = json.loads(conn.execute("SELECT data FROM runs WHERE id=1").fetchone()[0])
            task_data = json.loads(conn.execute("SELECT data FROM tasks WHERE id=1").fetchone()[0])
            fact_data = json.loads(conn.execute("SELECT value FROM facts WHERE key='durable_failure:demo'").fetchone()[0])
            failure_data = json.loads(conn.execute("SELECT data FROM failures WHERE id=1").fetchone()[0])
            ledger_value = conn.execute("SELECT value FROM life_ledger WHERE id=1").fetchone()[0]
        finally:
            conn.close()

        assert "persistent storage truncated" in run_data["output_json"]["summary"]
        assert run_data["output_json"]["items"][39]["_persistent_omitted_items"] == 11
        assert run_data["output_json"]["items"][-1]["index"] == 89
        assert "persistent storage truncated" in task_data["result_json"]["summary"]
        assert "last_summary" not in fact_data
        assert fact_data["last_summary_chars"] == len(huge)
        assert "durable failure summary truncated" in fact_data["last_summary_preview"]
        assert "persistent storage truncated" in failure_data["summary"]
        assert "life_ledger value truncated" in ledger_value


def test_compact_runtime_db_preserves_small_json_and_pending_chat():
    from core.maintenance import compact_runtime_db

    huge = "C" * 20000
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "runtime.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}')")
            conn.execute("CREATE TABLE facts (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '', scope TEXT, updated_at TEXT)")
            conn.execute("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, role TEXT, content TEXT, session_id TEXT, status TEXT)")
            conn.execute(
                "CREATE TABLE meta_reflections "
                "(id TEXT PRIMARY KEY, diagnosis TEXT, proposal TEXT, verification_plan TEXT)"
            )
            original_run_json = '{"b":2,"a":1}'
            original_fact_json = '{"count":1,"reason":"missing_path"}'
            conn.execute("INSERT INTO runs (id, data) VALUES (?, ?)", (1, original_run_json))
            conn.execute(
                "INSERT INTO facts (key, value, scope, updated_at) VALUES (?, ?, 'system', '')",
                ("durable_failure:small", original_fact_json),
            )
            conn.execute(
                "INSERT INTO chat_messages (id, role, content, session_id, status) VALUES (?, 'user', ?, '', 'pending')",
                (1, huge),
            )
            conn.execute(
                "INSERT INTO chat_messages (id, role, content, session_id, status) VALUES (?, 'assistant', ?, '', 'delivered')",
                (2, huge),
            )
            conn.execute(
                "INSERT INTO meta_reflections (id, diagnosis, proposal, verification_plan) VALUES (?, ?, ?, ?)",
                ("mr-1", huge, "short", "short"),
            )
            conn.commit()
        finally:
            conn.close()

        report = compact_runtime_db(db_path, apply=True)
        assert report["changed_rows"] == 2

        conn = sqlite3.connect(db_path)
        try:
            run_json = conn.execute("SELECT data FROM runs WHERE id=1").fetchone()[0]
            fact_json = conn.execute("SELECT value FROM facts WHERE key='durable_failure:small'").fetchone()[0]
            pending_chat = conn.execute("SELECT content FROM chat_messages WHERE id=1").fetchone()[0]
            delivered_chat = conn.execute("SELECT content FROM chat_messages WHERE id=2").fetchone()[0]
            diagnosis = conn.execute("SELECT diagnosis FROM meta_reflections WHERE id='mr-1'").fetchone()[0]
        finally:
            conn.close()

        assert run_json == original_run_json
        assert fact_json == original_fact_json
        assert pending_chat == huge
        assert "chat message truncated" in delivered_chat
        assert "meta_reflection diagnosis truncated" in diagnosis


def test_compact_memory_dir_dry_run_and_apply():
    from core.maintenance import compact_memory_dir

    huge = "D" * 20000 + "TAIL"
    with tempfile.TemporaryDirectory() as d:
        memory_dir = Path(d) / "memory"
        nodes = memory_dir / "nodes"
        archive = memory_dir / "archive"
        archive_by_prefix = memory_dir / "archive_by_prefix" / "run"
        nodes.mkdir(parents=True)
        archive.mkdir(parents=True)
        archive_by_prefix.mkdir(parents=True)

        small = '{"b":2,"a":1}'
        small_path = nodes / "small.json"
        archive_path = archive / "run-result-big.json"
        prefixed_path = archive_by_prefix / "wm-promoted-big.json"
        jsonl_path = archive / "run-results.jsonl"
        small_path.write_text(small, encoding="utf-8")
        archive_path.write_text(json.dumps({"summary": huge}, ensure_ascii=False), encoding="utf-8")
        prefixed_path.write_text(json.dumps({"body": huge}, ensure_ascii=False), encoding="utf-8")
        jsonl_path.write_text(
            "\n".join([
                json.dumps({"summary": huge}, ensure_ascii=False),
                json.dumps({"summary": "small"}, ensure_ascii=False),
            ]) + "\n",
            encoding="utf-8",
        )

        dry = compact_memory_dir(memory_dir)
        assert dry["dry_run"] is True
        assert dry["changed_files"] == 3
        assert archive_path.read_text(encoding="utf-8") == json.dumps({"summary": huge}, ensure_ascii=False)

        applied = compact_memory_dir(memory_dir, apply=True)
        assert applied["dry_run"] is False
        assert applied["changed_files"] == 3

        small_after = small_path.read_text(encoding="utf-8")
        archive_after = json.loads(archive_path.read_text(encoding="utf-8"))
        prefixed_after = json.loads(prefixed_path.read_text(encoding="utf-8"))
        jsonl_after = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert small_after == small
        assert "persistent storage truncated" in archive_after["summary"]
        assert archive_after["summary"].endswith("TAIL")
        assert "persistent storage truncated" in prefixed_after["body"]
        assert "persistent storage truncated" in jsonl_after[0]["summary"]
        assert jsonl_after[0]["summary"].endswith("TAIL")
        assert jsonl_after[1]["summary"] == "small"


def test_compact_memory_dir_compacts_semantic_and_episodic_dbs():
    from core.maintenance import compact_memory_dir

    huge = "E" * 20000 + "TAIL"
    with tempfile.TemporaryDirectory() as d:
        memory_dir = Path(d) / "memory"
        memory_dir.mkdir(parents=True)
        semantic_db = memory_dir / "semantic.db"
        episodic_db = memory_dir / "episodic.db"

        conn = sqlite3.connect(semantic_db)
        try:
            conn.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, title TEXT, body TEXT, tags TEXT)")
            conn.execute("CREATE VIRTUAL TABLE nodes_fts USING fts5(id UNINDEXED, title, body, tags)")
            conn.execute(
                "INSERT INTO nodes (id, title, body, tags) VALUES ('n1', 'short', ?, '[]')",
                (huge,),
            )
            conn.execute(
                "INSERT INTO nodes_fts(id, title, body, tags) VALUES ('n1', 'short', ?, '[]')",
                (huge,),
            )
            conn.commit()
        finally:
            conn.close()

        conn = sqlite3.connect(episodic_db)
        try:
            conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, data TEXT NOT NULL DEFAULT '{}')")
            conn.execute(
                "CREATE TABLE narrative "
                "(id INTEGER PRIMARY KEY, task_id TEXT, chat_id TEXT, interlocutor_id TEXT, role TEXT, content TEXT)"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE narrative_fts USING fts5("
                "id UNINDEXED, task_id, chat_id, interlocutor_id, role, content)"
            )
            conn.execute(
                "INSERT INTO events (id, data) VALUES (1, ?)",
                (json.dumps({"content": huge}, ensure_ascii=False),),
            )
            conn.execute(
                "INSERT INTO narrative (id, task_id, chat_id, interlocutor_id, role, content) "
                "VALUES (1, 'task-1', '', '', 'consolidation', ?)",
                (huge,),
            )
            conn.execute(
                "INSERT INTO narrative_fts(id, task_id, chat_id, interlocutor_id, role, content) "
                "VALUES (1, 'task-1', '', '', 'consolidation', ?)",
                (huge,),
            )
            conn.commit()
        finally:
            conn.close()

        dry = compact_memory_dir(memory_dir)
        assert dry["dry_run"] is True
        assert dry["changed_rows"] == 3

        conn = sqlite3.connect(semantic_db)
        try:
            assert conn.execute("SELECT body FROM nodes WHERE id='n1'").fetchone()[0] == huge
        finally:
            conn.close()

        applied = compact_memory_dir(memory_dir, apply=True)
        assert applied["dry_run"] is False
        assert applied["changed_rows"] == 3
        assert applied["dbs"]["semantic.db"]["changed_rows"] == 1
        assert applied["dbs"]["episodic.db"]["changed_rows"] == 2

        conn = sqlite3.connect(semantic_db)
        try:
            semantic_body = conn.execute("SELECT body FROM nodes WHERE id='n1'").fetchone()[0]
            semantic_fts_body = conn.execute("SELECT body FROM nodes_fts WHERE id='n1'").fetchone()[0]
        finally:
            conn.close()
        conn = sqlite3.connect(episodic_db)
        try:
            event_data = json.loads(conn.execute("SELECT data FROM events WHERE id=1").fetchone()[0])
            narrative_content = conn.execute("SELECT content FROM narrative WHERE id=1").fetchone()[0]
            narrative_fts_content = conn.execute("SELECT content FROM narrative_fts WHERE id=1").fetchone()[0]
        finally:
            conn.close()

        assert "semantic body truncated" in semantic_body
        assert semantic_body == semantic_fts_body
        assert "persistent storage truncated" in event_data["content"]
        assert "episodic narrative truncated" in narrative_content
        assert narrative_content == narrative_fts_content


def test_compact_memory_dir_compacts_episodic_markdown_files():
    from core.maintenance import compact_memory_dir

    huge = "---\nold block\n" + "M" * 20000 + "\nTAIL"
    with tempfile.TemporaryDirectory() as d:
        memory_dir = Path(d) / "memory"
        archive_dir = memory_dir / "episodic" / "archive"
        archive_dir.mkdir(parents=True)
        md_path = archive_dir / "task-1622.md"
        md_path.write_text(huge, encoding="utf-8")

        dry = compact_memory_dir(memory_dir)
        assert dry["dry_run"] is True
        assert dry["changed_files"] == 1
        assert dry["dirs"]["episodic"]["changed_files"] == 1
        assert md_path.read_text(encoding="utf-8") == huge

        applied = compact_memory_dir(memory_dir, apply=True)
        assert applied["dry_run"] is False
        assert applied["changed_files"] == 1

        compacted = md_path.read_text(encoding="utf-8")
        assert len(compacted) < len(huge)
        assert "episodic markdown truncated" in compacted
        assert compacted.endswith("TAIL")
