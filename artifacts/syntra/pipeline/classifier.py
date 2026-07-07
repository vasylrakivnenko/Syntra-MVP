"""Classifier — one LLM call per clause; temperature 0, structured output."""
from __future__ import annotations
import json
from models import Clause, Classification

TAXONOMY = [
    "limitation_of_liability",
    "indemnification",
    "term_and_termination",
    "confidentiality",
    "ip_ownership",
    "payment_terms",
    "governing_law",
    "auto_renewal",
    "data_protection",
    "assignment",
    "warranties",
    "dispute_resolution",
    "insurance",
    "non_solicitation",
    "other",
]


class Classifier:
    TAXONOMY = TAXONOMY

    def classify(self, clause: Clause) -> Classification:
        from llm import get_client, MODEL, llm_available

        if not llm_available():
            return Classification(clause_type="other", confidence=0.1, spans=[])

        client = get_client()
        prompt = (
            f"Classify this contract clause into exactly one category from the list below.\n\n"
            f"CATEGORIES: {', '.join(TAXONOMY)}\n\n"
            f"CLAUSE:\n{clause.text[:900]}\n\n"
            'Return JSON: {"clause_type":"<category>","confidence":0.0,"reasoning":"<one sentence>"}'
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            r = json.loads(resp.choices[0].message.content)
            clause_type = r.get("clause_type", "other")
            if clause_type not in TAXONOMY:
                clause_type = "other"
            return Classification(
                clause_type=clause_type,
                confidence=float(r.get("confidence", 0.5)),
                spans=[],
            )
        except Exception as exc:
            print(f"[classifier] error: {exc}")
            return Classification(clause_type="other", confidence=0.1, spans=[])
