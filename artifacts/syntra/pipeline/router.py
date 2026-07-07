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
        reason = build_escalation_reason(
            len(unacceptable), len(silences), len(abstains)
        )
        return [QueueItem(doc_id=doc_id, priority=priority, status="pending",
                          reason=reason)]


def build_escalation_reason(n_unacceptable: int, n_silences: int,
                            n_abstains: int) -> str:
    """Human-readable explanation of why a contract was escalated."""
    parts: list[str] = []
    if n_unacceptable:
        parts.append(f"{n_unacceptable} clause{'s' if n_unacceptable != 1 else ''} "
                     f"breach{'es' if n_unacceptable == 1 else ''} walk-away positions")
    if n_silences:
        parts.append(f"{n_silences} required clause{'s are' if n_silences != 1 else ' is'} missing")
    if n_abstains >= _ABSTAIN_ESCALATION_THRESHOLD:
        parts.append(f"{n_abstains} clause{'s' if n_abstains != 1 else ''} "
                     f"couldn't be assessed against the playbook")
    return "; ".join(parts)
