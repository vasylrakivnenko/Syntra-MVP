"""PlaybookEditor — manual + ask-AI editing of the service × policy matrix."""
from __future__ import annotations
import re
import json
import datetime
import yaml
from pathlib import Path
from models import Playbook, PolicyColumn, ServiceLine, Position

_DEFAULT_YAML = Path(__file__).parent / "default.yaml"


class PlaybookEditor:
    def __init__(self, playbook: Playbook):
        self.playbook = playbook

    # ── load / save ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "PlaybookEditor":
        """Load the latest saved playbook; fall back to default.yaml."""
        from database import get_db
        with get_db() as conn:
            row = conn.execute(
                "SELECT yaml_content FROM playbooks ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            data = yaml.safe_load(row["yaml_content"])
            return cls(Playbook(**data))
        with open(_DEFAULT_YAML) as f:
            data = yaml.safe_load(f)
        return cls(Playbook(**data))

    def save(self, user_id: str) -> str:
        from database import get_db
        version = datetime.datetime.utcnow().strftime("%Y-%m-%d.%H%M%S")
        self.playbook = Playbook(**{**self.playbook.model_dump(), "version": version})
        yaml_content = yaml.dump(self.playbook.model_dump(), allow_unicode=True, sort_keys=False)
        with get_db() as conn:
            conn.execute(
                "INSERT INTO playbooks (version, yaml_content, created_by, created_at) VALUES (?,?,?,?)",
                (version, yaml_content, user_id, datetime.datetime.utcnow().isoformat()),
            )
        return version

    # ── cell edits ───────────────────────────────────────────────────────────

    def edit_cell(self, service_line_id: str, policy_id: str, data: dict) -> None:
        sl = self.playbook.get_service_line(service_line_id)
        if sl is None:
            return
        existing = sl.positions.get(policy_id)
        cell_id = existing.id if existing else f"{service_line_id[:4].upper()}-{policy_id[:4].upper()}-{len(sl.positions)+1}"
        sl.positions[policy_id] = Position(
            id=cell_id,
            preferred=data.get("preferred", ""),
            fallback=data.get("fallback", ""),
            walk_away=data.get("walk_away", ""),
            source_doc_ids=existing.source_doc_ids if existing else [],
        )

    def promote_fallback(self, service_line_id: str, clause_type: str, text: str) -> None:
        """Attorney promotion: accepted clause text becomes the new fallback."""
        policy = self.playbook.get_policy_by_clause_type(clause_type)
        if policy is None:
            return
        sl = self.playbook.get_service_line(service_line_id)
        if sl is None:
            return
        cell = sl.positions.get(policy.id)
        if cell:
            cell.fallback = text[:400]

    # ── row / column management ───────────────────────────────────────────────

    def add_row(self, label: str, side: str) -> None:
        id_ = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        if not self.playbook.get_service_line(id_):
            self.playbook.service_lines.append(ServiceLine(id=id_, label=label, side=side))

    def delete_row(self, service_line_id: str) -> None:
        self.playbook.service_lines = [
            sl for sl in self.playbook.service_lines if sl.id != service_line_id
        ]

    def rename_row(self, service_line_id: str, new_label: str) -> None:
        sl = self.playbook.get_service_line(service_line_id)
        if sl:
            sl.label = new_label

    def add_column(self, label: str, clause_type: str) -> None:
        id_ = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        if not any(p.id == id_ for p in self.playbook.policies):
            self.playbook.policies.append(PolicyColumn(id=id_, label=label, clause_type=clause_type))

    def delete_column(self, policy_id: str) -> None:
        self.playbook.policies = [p for p in self.playbook.policies if p.id != policy_id]
        for sl in self.playbook.service_lines:
            sl.positions.pop(policy_id, None)

    def rename_column(self, policy_id: str, new_label: str) -> None:
        policy = next((p for p in self.playbook.policies if p.id == policy_id), None)
        if policy:
            policy.label = new_label

    # ── ask-AI editing ───────────────────────────────────────────────────────

    def edit_with_ai(self, prompt: str) -> list[str]:
        """Apply an AI-suggested diff to the matrix; return list of change descriptions."""
        from llm import get_client, MODEL
        client = get_client()
        matrix_yaml = yaml.dump(self.playbook.model_dump(), allow_unicode=True, sort_keys=False)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal contract expert. Given a playbook matrix in YAML and a user request, "
                        "return JSON with keys: 'changes' (array of human-readable change descriptions, max 5) "
                        "and 'updated_yaml' (the complete revised YAML string). Make minimal, targeted changes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Current playbook:\n{matrix_yaml}\n\nRequest: {prompt}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        result = json.loads(response.choices[0].message.content)
        updated_yaml = result.get("updated_yaml", matrix_yaml)
        try:
            self.playbook = Playbook(**yaml.safe_load(updated_yaml))
        except Exception:
            pass
        return result.get("changes", [])
