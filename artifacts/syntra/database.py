import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path

BASE = Path(__file__).parent
DB_PATH = BASE / "syntra.db"
SEED_PATH = BASE / "seed.json"


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # The bg pipeline thread now appends per-LLM-call audit events concurrently
    # with request-thread writes; without a busy timeout SQLite raises
    # "database is locked" immediately instead of waiting.
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   TEXT PRIMARY KEY,
            username  TEXT,
            role      TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS documents (
            doc_id       TEXT PRIMARY KEY,
            source_type  TEXT,
            status       TEXT,
            side         TEXT,
            service_line TEXT,
            uploaded_by  TEXT,
            uploaded_at  TEXT,
            filename     TEXT,
            content_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS clauses (
            clause_id    TEXT PRIMARY KEY,
            doc_id       TEXT,
            text         TEXT,
            start        INTEGER,
            end          INTEGER,
            heading_path TEXT
        );
        CREATE TABLE IF NOT EXISTS classifications (
            clause_id   TEXT PRIMARY KEY,
            clause_type TEXT,
            confidence  REAL,
            spans       TEXT
        );
        CREATE TABLE IF NOT EXISTS verdicts (
            clause_id    TEXT PRIMARY KEY,
            doc_id       TEXT,
            branch       TEXT,
            status       TEXT,
            rule_ids     TEXT,
            rationale    TEXT,
            reason       TEXT,
            service_line TEXT,
            risk_weight  INTEGER DEFAULT 3,
            suggested_text TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS rule_verdicts (
            id           TEXT PRIMARY KEY,
            doc_id       TEXT,
            rule_id      TEXT,
            policy_id    TEXT,
            clause_type  TEXT,
            verdict      TEXT,
            rationale    TEXT,
            question     TEXT,
            suggested_text TEXT DEFAULT '',
            evidence_clause_ids TEXT,
            cited_position TEXT,
            risk_weight  INTEGER DEFAULT 3
        );
        CREATE TABLE IF NOT EXISTS market_reports (
            doc_id         TEXT PRIMARY KEY,
            schema_version TEXT,
            report_json    TEXT,
            created_at     TEXT
        );
        CREATE TABLE IF NOT EXISTS queue_items (
            item_id       TEXT PRIMARY KEY,
            doc_id        TEXT,
            priority      INTEGER,
            assignee      TEXT,
            status        TEXT,
            reason        TEXT,
            created_at    TEXT,
            attorney_notes TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id     TEXT PRIMARY KEY,
            ts           TEXT,
            actor_id     TEXT,
            action       TEXT,
            input_hash   TEXT,
            prompt_hash  TEXT,
            output_json  TEXT,
            citations    TEXT,
            prev_hash    TEXT
        );
        CREATE TABLE IF NOT EXISTS playbooks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            version      TEXT,
            yaml_content TEXT,
            created_by   TEXT,
            created_at   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_verdicts_doc_id ON verdicts(doc_id);
        CREATE INDEX IF NOT EXISTS idx_rule_verdicts_doc_id ON rule_verdicts(doc_id);
        CREATE INDEX IF NOT EXISTS idx_queue_items_doc_id ON queue_items(doc_id);
        CREATE INDEX IF NOT EXISTS idx_queue_items_status ON queue_items(status);
        """)
        # Lightweight migrations for databases created before a column existed.
        for stmt in (
            "ALTER TABLE documents ADD COLUMN content_hash TEXT",
            "ALTER TABLE documents ADD COLUMN parties_json TEXT",
            "ALTER TABLE documents ADD COLUMN our_party TEXT",
            "ALTER TABLE documents ADD COLUMN urgency TEXT",
            "ALTER TABLE documents ADD COLUMN needed_by TEXT",
            "ALTER TABLE documents ADD COLUMN case_id TEXT",
            "ALTER TABLE documents ADD COLUMN version INTEGER",
            "ALTER TABLE queue_items ADD COLUMN reviewed_at TEXT",
            "ALTER TABLE queue_items ADD COLUMN reviewed_by TEXT",
            "ALTER TABLE queue_items ADD COLUMN acknowledged_at TEXT",
            "ALTER TABLE verdicts ADD COLUMN cited_position TEXT",
            "ALTER TABLE verdicts ADD COLUMN abstain_kind TEXT",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
        # Versioning backfill (idempotent): pre-versioning rows each become
        # their own single-version case.
        conn.execute(
            "UPDATE documents SET case_id=doc_id, version=1 WHERE case_id IS NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id)"
        )


def seed_from_file():
    if not SEED_PATH.exists():
        return
    with open(SEED_PATH) as f:
        data = json.load(f)
    with get_db() as conn:
        for u in data.get("users", []):
            conn.execute(
                "INSERT OR IGNORE INTO users VALUES (?,?,?,?)",
                (u["user_id"], u["username"], u["role"], u["created_at"])
            )
        for pb in data.get("playbooks", []):
            conn.execute(
                "INSERT OR IGNORE INTO playbooks (version, yaml_content, created_by, created_at) VALUES (?,?,?,?)",
                (pb["version"], pb["yaml_content"], pb.get("created_by", ""), pb.get("created_at", ""))
            )


def dump_to_file():
    with get_db() as conn:
        users = [dict(r) for r in conn.execute("SELECT * FROM users")]
        playbooks = [dict(r) for r in conn.execute("SELECT * FROM playbooks ORDER BY id")]
    data = {"users": users, "playbooks": playbooks}
    with open(SEED_PATH, "w") as f:
        json.dump(data, f, indent=2)
