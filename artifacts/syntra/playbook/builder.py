"""PlaybookBuilder — bootstrap a service × policy matrix from existing contracts."""
from __future__ import annotations
import json
import datetime
from models import Document, Playbook, PolicyColumn, ServiceLine, Position


_DEFAULT_POLICIES = [
    PolicyColumn(id="payment_terms",        label="Payment terms",         clause_type="payment_terms",         required=True,  risk_weight=4),
    PolicyColumn(id="limitation_of_liability", label="Limitation of liability", clause_type="limitation_of_liability", required=True,  risk_weight=5),
    PolicyColumn(id="term_and_termination", label="Term & termination",    clause_type="term_and_termination",  required=False, risk_weight=3),
    PolicyColumn(id="confidentiality",      label="Confidentiality",       clause_type="confidentiality",       required=False, risk_weight=3),
    PolicyColumn(id="indemnification",      label="Indemnification",       clause_type="indemnification",       required=False, risk_weight=4),
]


class PlaybookBuilder:
    def build(self, contracts: list[Document]) -> Playbook:
        """Two-pass bootstrap: segment clusters → infer positions per cell."""
        if not contracts:
            from playbook.editor import PlaybookEditor
            return PlaybookEditor.load().playbook

        from llm import get_client, MODEL
        from pipeline.segmenter import Segmenter

        client = get_client()
        segmenter = Segmenter()

        # Pass 1: cluster by (side, service_line)
        clusters: dict[tuple[str, str], list[Document]] = {}
        for doc in contracts:
            seg = segmenter.segment(doc)
            key = (seg.side, seg.service_line)
            clusters.setdefault(key, []).append(doc)

        service_lines: list[ServiceLine] = []

        # Pass 2: infer positions per cluster × policy
        for (side, sl_id), docs in clusters.items():
            combined = "\n\n---\n\n".join(d.full_text[:3000] for d in docs[:5])
            positions: dict[str, Position] = {}

            for policy in _DEFAULT_POLICIES:
                prompt = (
                    f"Extract the company's {policy.label} position from these contracts.\n\n"
                    f"CONTRACTS:\n{combined[:4500]}\n\n"
                    "Return JSON with:\n"
                    '- "found": true|false\n'
                    '- "preferred": most common position as brief text\n'
                    '- "fallback": acceptable variant\n'
                    '- "walk_away": clearly rejected position'
                )
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0,
                    )
                    r = json.loads(resp.choices[0].message.content)
                    if r.get("found", True):
                        positions[policy.id] = Position(
                            id=f"{sl_id[:4].upper()}-{policy.id[:4].upper()}-1",
                            preferred=r.get("preferred", ""),
                            fallback=r.get("fallback", ""),
                            walk_away=r.get("walk_away", ""),
                        )
                except Exception:
                    pass

            label = sl_id.replace("_", " ").title()
            service_lines.append(ServiceLine(id=sl_id, label=label, side=side, positions=positions))

        return Playbook(
            version=datetime.datetime.utcnow().strftime("%Y-%m-%d.%H%M%S"),
            policies=_DEFAULT_POLICIES,
            service_lines=service_lines,
        )
