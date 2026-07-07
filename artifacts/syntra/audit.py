"""Append-only, hash-chained audit log (§12)."""
import hashlib
import json
import datetime
from models import AuditEvent, Citation
from database import get_db


class AuditLog:
    def append(self, event: AuditEvent) -> AuditEvent:
        with get_db() as conn:
            last = conn.execute(
                "SELECT event_id, ts, actor_id, action, input_hash, "
                "prompt_hash, output_json, citations, prev_hash "
                "FROM audit_events ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            if last:
                prev_data = json.dumps(dict(last), sort_keys=True)
                event.prev_hash = hashlib.sha256(prev_data.encode()).hexdigest()
            conn.execute(
                "INSERT INTO audit_events VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    event.event_id,
                    event.ts,
                    event.actor_id,
                    event.action,
                    event.input_hash,
                    event.prompt_hash,
                    event.output_json,
                    json.dumps([c.model_dump() for c in event.citations]),
                    event.prev_hash,
                ),
            )
        return event

    def append_simple(self, actor_id: str, action: str, ref: str = "") -> AuditEvent:
        event = AuditEvent(
            ts=datetime.datetime.utcnow().isoformat(),
            actor_id=actor_id,
            action=action,
            input_hash=hashlib.sha256(ref.encode()).hexdigest()[:16],
        )
        return self.append(event)
