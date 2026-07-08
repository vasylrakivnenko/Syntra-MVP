---
name: Syntra dashboard risk column and attention sort
description: How the Risk column figures and the default table sort order are computed, for consistency if extended.
---

The Risk column (`risk_chip` macro in `_chips.html`) is driven by two correlated SQL subqueries added to a shared `_DOCS_QUERY` in `app.py`, used by both `index()` and `contracts()`:
- `risk_unacceptable`: COUNT of verdicts with `status='unacceptable'`.
- `risk_flags`: COUNT of verdicts where `(branch='verdict' AND status!='complies') OR branch='silence'`.

**Why:** These must mirror the `issues`/`silences`/`unacceptable` Jinja-side sets already computed in `contract.html`'s risk summary — any change to that page's risk semantics must be mirrored here or the table and detail page will disagree on what counts as a flag.

**How to apply:** If verdict branches/statuses are ever extended (e.g. a new `status` value), update both the SQL subqueries and `contract.html`'s Jinja `selectattr` filters together.

Default dashboard/contracts sort is "needs attention" (tiers: pending-review-or-urgent → high-urgency-unresolved → other-unresolved → approved/rejected), computed in Python (`_attention_sort_key`) after fetching all docs — NOT in SQL. A `?sort=recent` query param falls back to the plain `ORDER BY uploaded_at DESC`. Added indexes on `verdicts(doc_id)`, `queue_items(doc_id)`, `queue_items(status)` to keep the correlated subqueries cheap as data grows.
