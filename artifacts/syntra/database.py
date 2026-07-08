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
        """)
        # Lightweight migrations for databases created before a column existed.
        for stmt in (
            "ALTER TABLE documents ADD COLUMN content_hash TEXT",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists


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
