"""Router — decide which contracts need attorney review and build QueueItems."""
from __future__ import annotations
from models import ClauseVerdict, QueueItem, RuleVerdict

_ABSTAIN_ESCALATION_THRESHOLD = 3   # legacy path: escalate if >= N abstains


class Router:
    def route(self, doc_id: str, verdicts: list[ClauseVerdict],
              rule_verdicts: list[RuleVerdict] | None = None) -> list[QueueItem]:
        if rule_verdicts:
            return self._route_rules(doc_id, rule_verdicts)
        return self._route_legacy(doc_id, verdicts)

    # ── rule-level routing (reconciled documents) ────────────────────────────
    def _route_rules(self, doc_id: str,
                     rule_verdicts: list[RuleVerdict]) -> list[QueueItem]:
        breaches  = [r for r in rule_verdicts if r.verdict == "breach"]
        gaps      = [r for r in rule_verdicts if r.verdict == "not_covered"]
        questions = [r for r in rule_verdicts if r.verdict == "attorney_question"]
        fallbacks = [r for r in rule_verdicts if r.verdict == "met_via_fallback"]
        # outside_playbook is an FYI, never a routing signal

        if not (breaches or gaps or questions):
            return []

        # Priority stays on the same scale the queue UI's risk_label expects.
        priority = (
            len(breaches) * 5
            + len(gaps) * 3
            + len(fallbacks) * 1
            + (2 if questions else 0)
        )
        reason = build_rule_escalation_reason(
            len(breaches), len(gaps), len(questions)
        )
        return [QueueItem(doc_id=doc_id, priority=priority, status="pending",
                          reason=reason)]

    # ── legacy per-clause routing (pre-reconciliation docs, no-key mode) ─────
    def _route_legacy(self, doc_id: str,
                      verdicts: list[ClauseVerdict]) -> list[QueueItem]:
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


def build_rule_escalation_reason(n_breaches: int, n_gaps: int,
                                 n_questions: int) -> str:
    """Human-readable explanation of a rule-level escalation."""
    parts: list[str] = []
    if n_breaches:
        parts.append(f"{n_breaches} playbook position{'s are' if n_breaches != 1 else ' is'} "
                     "breached")
    if n_gaps:
        parts.append(f"{n_gaps} required polic{'ies have' if n_gaps != 1 else 'y has'} "
                     "no position defined")
    if n_questions:
        parts.append(f"{n_questions} question{'s' if n_questions != 1 else ''} "
                     "for attorney review")
    return "; ".join(parts)


def build_escalation_reason(n_unacceptable: int, n_silences: int,
                            n_abstains: int) -> str:
    """Human-readable explanation of why a contract was escalated (legacy)."""
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
