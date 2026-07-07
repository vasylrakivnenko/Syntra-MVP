---
name: syntra playbook lookup coupling
description: Why the Segmenter's service_line must be a real playbook row id, and how lookup fallback works.
---

# Segment ↔ playbook coupling (Syntra triage)

Syntra's triage is a 2-D matrix lookup: `Playbook.lookup(segment.service_line, clause_type)`. The
`segment.service_line` MUST equal an existing `ServiceLine.id` in the playbook
(`general_supplier` / `general_customer` / `nda_standalone`, plus any user-added rows).

**Failure mode (the "everything abstains" bug):** the Segmenter's LLM used to invent a free-form
snake_case id (e.g. `nda`). No row has that id, so `lookup` returned None for *every* clause and the
whole document came back "0 flagged, N abstained" — even for blatantly unacceptable clauses.

**Fixes in place:**
- The Segmenter is passed the playbook and must choose an id from the enumerated rows; a guardrail
  maps any unknown id to a *real* row for the detected side (never a hardcoded id that might not
  exist), so a playbook without general rows cannot reintroduce universal-abstain.
- `Playbook.lookup` falls back to the `general_<side>` row when a specialised row (e.g. NDA) lacks a
  position for a clause type, so broad clause types (liability, indemnification, IP) stay comparable.
  The cited rule id is always the rule actually applied, so citations stay truthful.

**How to apply / watch for:**
- Adding a new clause type to the classifier taxonomy does nothing unless a matching `PolicyColumn`
  (by `clause_type`) exists AND some row defines a position — otherwise it silently abstains.
- A policy column with no position in any row can never produce a verdict (orphan). Keep columns and
  positions in sync.
- `nda_standalone` is fixed `side: supplier`; the Segmenter overwrites LLM side with the row's side,
  so a customer-side NDA would route to supplier rules. Low impact for a mutual-NDA taxonomy, but the
  one place a wrong-side citation can occur.
