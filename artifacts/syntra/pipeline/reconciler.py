"""Reconciler — one verdict per playbook rule, evidence cited across clauses.

Runs AFTER per-clause triage and BEFORE the redliner/router. Per-clause
verdicts stay persisted as the evidence layer; this stage fixes the
aggregation failure where triage hedges on a clause ("term length not
stated here") while the answer sits in a different clause of the same
document.

Cost discipline: rules whose clauses agree and never hedged are mapped
deterministically (zero LLM calls). One audited LLM call per rule ONLY when
clauses hedged (abstain/unusual) or conflict — typically 1-4 calls per doc.

Failure discipline: if the model is unavailable this stage returns nothing
and consumers fall back to per-clause rendering; if a single reconcile call
fails, the rule degrades to a deterministic worst-of aggregation and
unresolved hedges become a bundled attorney question. Nothing is ever lost —
clause verdicts are written before this stage runs.
"""
from __future__ import annotations
import json
from models import Clause, ClauseVerdict, Playbook, RuleVerdict, Segment

_LOW_CONFIDENCE = 0.40
_MAX_LLM_CALLS = 6          # per document; beyond this, deterministic fallback
_MAX_CLAUSE_CHARS = 1200    # per clause of document context in the prompt
_MAX_DOC_CLAUSES = 25
_DOC_QUALITY_THRESHOLD = 8  # > N unattached hedges = document-quality problem

_STATUS_TO_VERDICT = {
    "complies": "met",
    "acceptable_deviation": "met_via_fallback",
    "unacceptable": "breach",
}
# worst-of ordering for deterministic degradation
_SEVERITY = {"breach": 3, "met_via_fallback": 2, "met": 1}


class Reconciler:
    def reconcile(
        self,
        doc_id: str,
        verdicts: list[ClauseVerdict],
        clauses: list[Clause],
        clause_types: dict[str, str],
        segment: Segment,
        playbook: Playbook,
    ) -> list[RuleVerdict]:
        from llm import llm_available

        if not llm_available():
            return []  # no-key mode: legacy per-clause rendering stays in charge

        clause_map = {c.id: c for c in clauses}
        policies = {p.clause_type: p for p in playbook.policies}
        out: list[RuleVerdict] = []

        # ── partition clause verdicts ────────────────────────────────────────
        rule_groups: dict[str, dict] = {}   # rule_id -> {"decided": [], "hedged": []}
        outside: list[ClauseVerdict] = []
        loose_hedges: list[ClauseVerdict] = []

        def group(rule_id: str) -> dict:
            return rule_groups.setdefault(rule_id, {"decided": [], "hedged": []})

        for v in verdicts:
            if v.branch == "silence":
                policy = (v.cited_position or {}).get("policy_id")
                out.append(RuleVerdict(
                    doc_id=doc_id, rule_id=None, policy_id=policy,
                    clause_type=(v.cited_position or {}).get("clause_type"),
                    verdict="not_covered",
                    rationale=v.reason or "Required policy has no position for this service line.",
                    evidence_clause_ids=[v.clause_id],
                    cited_position=v.cited_position,
                    risk_weight=self._risk_weight(v, clause_types, policies),
                ))
            elif v.branch == "verdict":
                rid = (v.rule_ids or [None])[0]
                if rid is None:
                    continue
                bucket = "hedged" if v.status == "unusual" else "decided"
                group(rid)[bucket].append(v)
            elif v.branch == "abstain":
                if v.abstain_kind == "outside_playbook":
                    outside.append(v)
                else:  # low_confidence / llm_error
                    rid = (v.rule_ids or [None])[0]
                    if rid is not None:
                        group(rid)["hedged"].append(v)
                    else:
                        loose_hedges.append(v)

        # ── outside-playbook: FYI rows, never risk ───────────────────────────
        for v in outside:
            ctype = clause_types.get(v.clause_id)
            out.append(RuleVerdict(
                doc_id=doc_id, verdict="outside_playbook",
                clause_type=ctype,
                rationale=v.reason or "This clause type is outside the playbook.",
                evidence_clause_ids=[v.clause_id],
                risk_weight=0,
            ))

        # ── per-rule reconciliation ──────────────────────────────────────────
        llm_calls = 0
        for rule_id, g in rule_groups.items():
            decided, hedged = g["decided"], g["hedged"]
            statuses = {v.status for v in decided}
            members = decided + hedged
            cited = next((v.cited_position for v in members if v.cited_position), None)
            rw = max((self._risk_weight(v, clause_types, policies) for v in members),
                     default=3)
            base = dict(doc_id=doc_id, rule_id=rule_id,
                        policy_id=(cited or {}).get("policy_id"),
                        clause_type=(cited or {}).get("clause_type"),
                        cited_position=cited, risk_weight=rw)

            if not hedged and len(statuses) == 1:
                # clean pass-through: zero LLM calls
                top = max(decided, key=lambda v: _SEVERITY.get(
                    _STATUS_TO_VERDICT.get(v.status or "", ""), 0))
                out.append(RuleVerdict(
                    **base,
                    verdict=_STATUS_TO_VERDICT[next(iter(statuses))],
                    rationale=top.rationale,
                    suggested_text=top.suggested_text or "",
                    evidence_clause_ids=[v.clause_id for v in decided],
                ))
                continue

            rv = None
            if llm_calls < _MAX_LLM_CALLS:
                rv = self._reconcile_rule(base, rule_id, members, clauses,
                                          clause_map, cited)
                llm_calls += 1
            if rv is None:
                rv = self._worst_of(base, decided, hedged, clause_map)
            out.append(rv)

        # ── unattached hedges: one bundled question, never N queue items ─────
        if loose_hedges:
            out.append(self._bundle_loose(doc_id, loose_hedges, clause_map))

        return out

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _risk_weight(v: ClauseVerdict, clause_types: dict, policies: dict) -> int:
        ctype = clause_types.get(v.clause_id) or (v.cited_position or {}).get("clause_type")
        policy = policies.get(ctype)
        return policy.risk_weight if policy else 3

    def _reconcile_rule(self, base, rule_id, members, clauses, clause_map, cited):
        """One audited LLM call: judge the RULE against the whole document."""
        from llm import audited_chat, MODEL

        cp = cited or {}
        findings = "\n".join(
            f"- clause {v.clause_id}: "
            + (f"{v.status} — {v.rationale}" if v.branch == "verdict"
               else f"hedged — {v.reason}")
            for v in members
        )
        doc_text = "\n\n".join(
            f"[{c.id}] {' '.join(c.text.split())[:_MAX_CLAUSE_CHARS]}"
            for c in clauses[:_MAX_DOC_CLAUSES]
        )
        prompt = (
            "You are reconciling contract-review findings at the RULE level. "
            "Decide whether the DOCUMENT AS A WHOLE satisfies the company position. "
            "Information satisfying a requirement may appear in ANY clause, not just "
            "the clause where the issue was first noticed. Do not hedge about a "
            "requirement if another clause of this document plainly answers it.\n\n"
            f"COMPANY POSITION (rule {rule_id} — {cp.get('policy_label') or cp.get('clause_type') or 'policy'}):\n"
            f"  Preferred : {cp.get('preferred', '')}\n"
            f"  Fallback  : {cp.get('fallback', '')}\n"
            f"  Walk-away : {cp.get('walk_away', '')}\n\n"
            f"PER-CLAUSE FINDINGS SO FAR (may be over-hedged):\n{findings}\n\n"
            f"DOCUMENT CLAUSES:\n{doc_text}\n\n"
            "Return JSON:\n"
            '{"verdict":"met|met_via_fallback|breach|insufficient",'
            '"rationale":"<=2 plain-language sentences; refer to clauses by their '
            'numbering in the document text, never by internal ids",'
            '"evidence_clause_ids":["<ids in [brackets] above that support the verdict>"],'
            '"suggested_text":"<replacement language if breach, else empty>",'
            '"question":"<ONLY if insufficient: one specific question for the '
            'supervising attorney, referencing what is ambiguous>",'
            '"confidence":0.0}'
        )
        try:
            resp = audited_chat(
                "reconcile", ref=f"{base['doc_id']}:{rule_id}",
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            r = json.loads(resp.choices[0].message.content)
            verdict = r.get("verdict", "insufficient")
            confidence = float(r.get("confidence", 0.7))
            evidence = [cid for cid in (r.get("evidence_clause_ids") or [])
                        if cid in clause_map] or [v.clause_id for v in members]
            if verdict not in ("met", "met_via_fallback", "breach", "insufficient"):
                verdict = "insufficient"
            if verdict == "insufficient" or confidence < _LOW_CONFIDENCE:
                question = (r.get("question") or "").strip() or (
                    "The document's language on this position is ambiguous — "
                    "does it satisfy the company's requirements?"
                )
                return RuleVerdict(**base, verdict="attorney_question",
                                   rationale=r.get("rationale", ""),
                                   question=question,
                                   evidence_clause_ids=evidence)
            return RuleVerdict(**base, verdict=verdict,
                               rationale=r.get("rationale", ""),
                               suggested_text=r.get("suggested_text", "") or "",
                               evidence_clause_ids=evidence)
        except Exception:
            return None  # caller degrades deterministically

    @staticmethod
    def _worst_of(base, decided, hedged, clause_map):
        """Deterministic degradation when the reconcile call fails."""
        evidence = [v.clause_id for v in decided + hedged]
        if decided:
            worst = max(decided, key=lambda v: _SEVERITY.get(
                _STATUS_TO_VERDICT.get(v.status or "", ""), 0))
            verdict = _STATUS_TO_VERDICT.get(worst.status or "", "met_via_fallback")
            return RuleVerdict(**base, verdict=verdict,
                               rationale=worst.rationale,
                               suggested_text=worst.suggested_text or "",
                               evidence_clause_ids=evidence)
        # only hedges: this rule genuinely needs a human
        return RuleVerdict(
            **base, verdict="attorney_question",
            rationale="Automated review couldn't resolve this position from the document.",
            question="Automated review couldn't determine whether this document "
                     "satisfies the company position — please assess the cited clauses.",
            evidence_clause_ids=evidence,
        )

    @staticmethod
    def _bundle_loose(doc_id, hedges, clause_map):
        """All rule-less low-confidence clauses become ONE bundled question."""
        evidence = [v.clause_id for v in hedges]
        n = len(hedges)
        if n > _DOC_QUALITY_THRESHOLD:
            return RuleVerdict(
                doc_id=doc_id, verdict="attorney_question",
                rationale=f"{n} clauses couldn't be read reliably — likely a "
                          "document-quality problem (scan/formatting).",
                question=f"{n} clauses in this document couldn't be analyzed "
                         "reliably. Is the source file readable, and does a "
                         "manual skim raise any playbook concerns?",
                evidence_clause_ids=evidence, risk_weight=2,
            )
        return RuleVerdict(
            doc_id=doc_id, verdict="attorney_question",
            rationale=f"{n} clause{'s' if n != 1 else ''} couldn't be matched to "
                      "the playbook with confidence.",
            question=f"{n} clause{'s' if n != 1 else ''} couldn't be assessed "
                     "automatically — do the cited clauses raise any playbook "
                     "concerns?",
            evidence_clause_ids=evidence, risk_weight=2,
        )
