"""Router — decide which contracts need attorney review and build QueueItems."""
from __future__ import annotations
from models import ClauseVerdict, QueueItem

_ABSTAIN_ESCALATION_THRESHOLD = 3   # escalate if >= N abstains (high uncertainty)


class Router:
    def route(self, doc_id: str, verdicts: list[ClauseVerdict]) -> list[QueueItem]:
        unacceptable = [v for v in verdicts if v.branch == "verdict" and v.status == "unacceptable"]
        silences     = [v for v in verdicts if v.branch == "silence"]
        abstains     = [v for v in verdicts if v.branch == "abstain"]
        deviations   = [v for v in verdicts if v.branch == "verdict" and v.status == "acceptable_deviation"]

        needs_attorney = (
            len(unacceptable) > 0
            or len(silences) > 0
            or len(abstains) >= _ABSTAIN_ESCALATION_THRESHOLD
        )

        if not needs_attorney:
            return []

        priority = (
            len(unacceptable) * 5
            + len(silences) * 3
            + len(deviations) * 1
            + len(abstains)
        )
        return [QueueItem(doc_id=doc_id, priority=priority, status="pending")]
