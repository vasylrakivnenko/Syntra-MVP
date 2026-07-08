"""Triage — deterministic 2-D matrix lookup + one LLM compliance judgment (§8)."""
from __future__ import annotations
import json
from models import Clause, Classification, Segment, Playbook, Position, ClauseVerdict

_ABSTAIN_CONFIDENCE = 0.40
_SEGMENT_CONFIDENCE = 0.35


class Triage:
    def decide(
        self,
        clause: Clause,
        classification: Classification,
        segment: Segment,
        playbook: Playbook,
    ) -> ClauseVerdict:
        # 1. Low-confidence classification → abstain
        if classification.confidence < _ABSTAIN_CONFIDENCE:
            return ClauseVerdict(
                clause_id=clause.id, branch="abstain",
                reason=f"Low classification confidence ({classification.confidence:.2f})",
                service_line=segment.service_line,
            )

        # 2. Low-confidence segment → abstain
        if segment.confidence < _SEGMENT_CONFIDENCE:
            return ClauseVerdict(
                clause_id=clause.id, branch="abstain",
                reason="Could not reliably identify service line",
                service_line=segment.service_line,
            )

        # 3. "other" type is never actionable
        if classification.clause_type == "other":
            return ClauseVerdict(
                clause_id=clause.id, branch="abstain",
                reason="Clause classified as 'other'; no playbook coverage",
                service_line=segment.service_line,
            )

        # 4. Two-axis deterministic lookup
        position, resolved_sl = playbook.lookup_resolved(
            segment.service_line, classification.clause_type
        )
        policy = playbook.get_policy_by_clause_type(classification.clause_type)

        if position is None:
            if policy and policy.required:
                return ClauseVerdict(
                    clause_id=clause.id, branch="silence",
                    reason=(
                        f"Required policy '{classification.clause_type}' has no position "
                        f"defined for service line '{segment.service_line}'"
                    ),
                    service_line=segment.service_line,
                    cited_position={
                        "policy_id": policy.id,
                        "policy_label": policy.label,
                        "clause_type": classification.clause_type,
                        "service_line": segment.service_line,
                        "required": True,
                        "playbook_version": playbook.version,
                        "as_of": "analysis",
                    },
                )
            return ClauseVerdict(
                clause_id=clause.id, branch="abstain",
                reason=f"No playbook position for ({segment.service_line}, {classification.clause_type})",
                service_line=segment.service_line,
            )

        # Snapshot the exact position this clause is judged against, so the
        # citation survives later playbook edits (e.g. promote-to-fallback).
        cited_position = {
            "rule_id": position.id,
            "policy_id": policy.id if policy else None,
            "policy_label": policy.label if policy else classification.clause_type,
            "clause_type": classification.clause_type,
            "service_line": resolved_sl,
            "preferred": position.preferred,
            "fallback": position.fallback,
            "walk_away": position.walk_away,
            "playbook_version": playbook.version,
            "as_of": "analysis",
        }

        # 5. LLM compliance judgment
        return self._compare(clause, classification, position, segment, cited_position)

    def _compare(
        self,
        clause: Clause,
        classification: Classification,
        position: Position,
        segment: Segment,
        cited_position: dict | None = None,
    ) -> ClauseVerdict:
        from llm import audited_chat, MODEL, llm_available

        if not llm_available():
            return ClauseVerdict(
                clause_id=clause.id, branch="abstain",
                reason="LLM unavailable — set AI_INTEGRATIONS_OPENAI_API_KEY",
                service_line=segment.service_line,
            )

        prompt = (
            "You are a contract review expert. Compare this clause against the company's playbook position.\n\n"
            f"CLAUSE:\n{clause.text[:900]}\n\n"
            f"COMPANY POSITION (rule {position.id}):\n"
            f"  Preferred : {position.preferred}\n"
            f"  Fallback  : {position.fallback}\n"
            f"  Walk-away : {position.walk_away}\n\n"
            "Return JSON:\n"
            '{"status":"complies|acceptable_deviation|unacceptable|unusual",'
            '"rationale":"<one sentence citing the rule>",'
            '"confidence":0.0,'
            '"suggested_text":"<brief suggested alternative if unacceptable, else empty>"}'
        )
        try:
            resp = audited_chat(
                "triage", ref=clause.id,
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            r = json.loads(resp.choices[0].message.content)
            status = r.get("status", "unusual")
            confidence = float(r.get("confidence", 0.7))

            if confidence < _ABSTAIN_CONFIDENCE:
                return ClauseVerdict(
                    clause_id=clause.id, branch="abstain",
                    reason="Insufficient grounding for compliance judgment",
                    service_line=segment.service_line,
                )

            valid_statuses = {"complies", "acceptable_deviation", "unacceptable", "unusual"}
            if status not in valid_statuses:
                status = "unusual"

            return ClauseVerdict(
                clause_id=clause.id,
                branch="verdict",
                status=status,
                rule_ids=[position.id],
                rationale=r.get("rationale", ""),
                suggested_text=r.get("suggested_text", ""),
                service_line=segment.service_line,
                cited_position=cited_position,
            )
        except Exception as exc:
            return ClauseVerdict(
                clause_id=clause.id, branch="abstain",
                reason=f"LLM error: {str(exc)[:120]}",
                service_line=segment.service_line,
            )
