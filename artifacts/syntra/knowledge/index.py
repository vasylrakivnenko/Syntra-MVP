"""KnowledgeIndex — BM25 retrieval over playbook positions (lane-two support, §5.2)."""
from __future__ import annotations
from typing import Any


class KnowledgeIndex:
    """Retrieves relevant playbook passages for a clause text.

    Falls back to a simple linear scan when rank_bm25 is not installed.
    ContextHub (github.com/andrewyng/context-hub) can replace _bm25 via
    the same interface for production use.
    """

    def __init__(self):
        self._corpus: list[dict[str, Any]] = []
        self._bm25: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._build_corpus()

    def _build_corpus(self) -> None:
        from playbook.editor import PlaybookEditor

        pb = PlaybookEditor.load().playbook
        corpus: list[dict[str, Any]] = []

        for sl in pb.service_lines:
            for policy_id, pos in sl.positions.items():
                policy = next((p for p in pb.policies if p.id == policy_id), None)
                policy_label = policy.label if policy else policy_id
                text = (
                    f"{sl.label} — {policy_label}: "
                    f"preferred={pos.preferred}; fallback={pos.fallback}"
                )
                corpus.append({
                    "text": text,
                    "rule_id": pos.id,
                    "service_line": sl.id,
                    "clause_type": policy.clause_type if policy else "",
                })

        self._corpus = corpus
        if not corpus:
            return

        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            tokenized = [c["text"].lower().split() for c in corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            pass

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        self._ensure_loaded()
        if not self._corpus:
            return []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
            return [self._corpus[i] for i in indices]
        # Linear fallback
        return self._corpus[:top_k]

    def invalidate(self) -> None:
        """Call after playbook edits so the index is rebuilt on next search."""
        self._loaded = False
        self._corpus = []
        self._bm25 = None
