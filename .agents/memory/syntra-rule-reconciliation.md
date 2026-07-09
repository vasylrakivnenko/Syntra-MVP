---
name: Syntra rule-level reconciliation
description: Invariants for the rule-verdict layer that sits on top of per-clause triage — dual-mode views, lockstep counting, and why outside_playbook never flags.
---

The verdict unit shown to users is the playbook rule, not the clause. Per-clause triage rows stay untouched underneath; a post-triage reconciler emits terminal rule verdicts (met / met_via_fallback / breach / attorney_question / not_covered / outside_playbook).

**Why:** clause-level output rendered hedges ("couldn't assess") as user-facing dead ends; operators need one answer per company position, and unresolved hedges belong with an attorney as a concrete question, not a shrug.

**How to apply:**
- Mode detection everywhere is "does this doc have rule_verdicts rows" — empty means legacy/no-key doc and every surface (contract page, dashboard risk chips, version diff counter) must fall back to per-clause logic. Never backfill old docs.
- Any new surface that counts findings must use the rule-level taxonomy when rule rows exist, and must treat `outside_playbook` as FYI-only — it never counts as a flag and never routes.
- Hedged rules cost one audited LLM call each (bounded per doc); clean rules must stay deterministic — do not add LLM calls for complies/unacceptable-only rules.
- Rule mode must never render "couldn't assess": every hedge terminates as a verdict, an attorney question, or an FYI.
- Nested-anchor gotcha: rule mode nests clause cards inside rule `<details>`, so citation reveal JS must open all ancestor details, and clauses cited by no rule still need anchor targets on the page.
