"""Lane-two agent — knowledge-grounded re-classification for low-confidence clauses (§5.2)."""
from __future__ import annotations
import json
from models import Clause, Classification
from knowledge.index import KnowledgeIndex
from pipeline.classifier import TAXONOMY

_LANE_TWO_THRESHOLD = 0.60   # only run when primary confidence is below this


class LaneTwoAgent:
    def __init__(self):
        self._index = KnowledgeIndex()

    def run(self, clause: Clause, primary: Classification) -> Classification | None:
        """Re-classify using playbook knowledge. Returns improved Classification or None."""
        if primary.confidence >= _LANE_TWO_THRESHOLD:
            return None   # primary is good enough; no need for lane two

        passages = self._index.search(clause.text[:300], top_k=4)
        if not passages:
            return None

        from llm import audited_chat, MODEL, llm_available
        if not llm_available():
            return None

        context = "\n".join(f"- {p['text']}" for p in passages)
        prompt = (
            "You are a contract review expert with access to the following company policy passages.\n\n"
            f"COMPANY KNOWLEDGE:\n{context}\n\n"
            f"CLAUSE:\n{clause.text[:700]}\n\n"
            f"Classify the clause into one of: {', '.join(TAXONOMY)}\n"
            'Return JSON: {"clause_type":"<category>","confidence":0.0}'
        )
        try:
            resp = audited_chat(
                "lane_two", ref=clause.id,
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            r = json.loads(resp.choices[0].message.content)
            clause_type = r.get("clause_type", primary.clause_type)
            if clause_type not in TAXONOMY:
                clause_type = primary.clause_type
            improved = Classification(
                clause_type=clause_type,
                confidence=float(r.get("confidence", primary.confidence)),
                spans=[],
            )
            if improved.confidence > primary.confidence:
                return improved
        except Exception as exc:
            print(f"[lane_two] error: {exc}")
        return None
