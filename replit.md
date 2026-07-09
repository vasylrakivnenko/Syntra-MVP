# [Syntra]

Syntra provides SMBs and mid-market companies with high quality legal support and cut legal review costs by 90% and deal cycles by 70% throughout AI-native general counsel product that ingests contracts or legal requests, compares them to company positions, drafts redlines and responses, flags risk, and routes high-risk items to a supervising attorney.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/syntra/` — Flask app (app.py routes, database.py schema, pipeline/ stages, templates/)
- `artifacts/syntra/market_lens/` — vendored NDA benchmarking lib v2 (do not edit); `market_data/` holds the combined 1,158-NDA table (ContractNLI + CUAD + MCC; market.sqlite + omx_reference.json + DATA_ATTRIBUTION.md)
- `artifacts/syntra/pipeline/market.py` — the only adapter between Syntra and market_lens (extraction via llm.py, offline scoring, LLM synthesis)
- `artifacts/syntra/pipeline/market_tabpfn.py` — optional TabPFN rarity signal; entirely inert unless TABPFN_TOKEN is set

## Architecture decisions

- Risk vs urgency are orthogonal: `queue_items.priority` is machine-computed risk (router weights: 5×walk-away breach + 3×missing clause + deviations + abstains, market-only escalations floored at 3); `documents.urgency`/`needed_by` are operator-declared at upload. Attorney triage sorts urgency → deadline → risk.
- Re-uploading an identical file (content-hash dedupe) is the intentional escape hatch to escalate urgency/deadline on an existing contract — it updates but never downgrades.
- AuditLog opens its own SQLite connection; audit calls must happen AFTER `with get_db()` write blocks commit, or SQLite locks ("database is locked").
- Templates share urgency/deadline chips via `templates/_chips.html` macros — edit there, not inline.
- Dashboard/Contracts tables use a Risk column (`risk_chip` macro) backed by two correlated SQL subqueries per doc in a shared `_DOCS_QUERY`; these must stay in lockstep with `contract.html`'s Jinja-side issues/silences/unacceptable risk sets — see `.agents/memory/syntra-dashboard-risk-sort.md`.
- Default table sort is "needs attention" (pending review/urgent → high urgency → other unresolved → resolved), computed Python-side after fetch, not in SQL; `?sort=recent` opts into plain upload-time order.
- Role-aware inbox bell is injected via `@app.context_processor` (attorney → pending queue items; operator → unacknowledged decisions on own uploads). Acknowledgment happens ONLY via the bell link (`?ack=1` on the contract page) — a plain page view/refresh must never clear the bell.
- Live updates are polling-based: base.html polls `/notifications.json` every 12s (rebuilds badge + dropdown via DOM APIs, textContent only); contract pages pending attorney review poll `/contracts/<id>/review-status` every 10s and reload when the decision lands. Endpoints must NOT live under `/api/*` — the workspace proxy routes that prefix to the separate API Server artifact.
- Grounded citations: triage snapshots the exact playbook position into `verdicts.cited_position` at analysis time (`as_of:"analysis"`) so citations survive playbook edits; rows analyzed before the feature are backfilled on read from the CURRENT playbook (`as_of:"current"`, labeled as such in the UI). Abstains are intentionally uncited.
- "Silence" verdicts mean the clause EXISTS but the playbook has no position for it (gap is in OUR playbook, not the document) — citation UIs must link the clause, not claim it's missing.
- MVP disclaimer: shared modal `#mvpModal` in base.html. Shown one-shot after uploads via `session["mvp_notice"]` popped by the `inject_mvp_notice` context processor; forced every visit on redline preview via `show_mvp_modal=True` in render_template (template context overrides processor); download links use class `mvp-guard` (JS click intercept). Navbar carries a `.mvp-sup` superscript badge.
- Full model-action audit: every LLM call routes through `llm.audited_chat(stage, ref, **kwargs)`, which appends a hash-chained `llm_call:<stage>` AuditEvent (sha256 of prompt + output, never raw text; audit failure never breaks the call). Actor comes from the `set_llm_actor` contextvar — set at the top of `_run_pipeline` (bg thread), before `infer_parties` in upload, and in `playbook_ai`; default "system". Playbook saves audit `playbook_<action>` / `playbook_ai_edit` AFTER `editor.save()` commits. `AuditLog.append` serializes via a module `threading.Lock` and picks the chain tip by `rowid DESC`; `get_db` sets `busy_timeout=5000` to absorb bg-thread write contention.
- Citation UI is shared via `templates/_citations.html` macro (`link_prefix` arg for cross-page use); playbook matrix cells carry unconditional `pos-<sl>-<policy>` anchors, clause cards carry `clause-<id>` anchors; the .docx redline embeds plain-text SOURCES blocks (no hyperlinks).
- Contract versioning: each upload row is immutable; `documents.case_id` groups versions (v1's case_id = its own doc_id, backfilled on migrate), `documents.version` increments via race-safe `COALESCE(MAX(version))+1` subquery INSERT. Pipeline stays 100% per-doc_id — zero pipeline changes. Dashboard/contracts `_DOCS_QUERY` shows latest-version-only.
- Version upload flow: mode radio on /upload (`mode=version` + `parent_case_id`); eligibility = case's latest version status IN ('processed','error') — mid-pipeline cases are not offered. Identical-file check in version mode compares hash against the case's own latest only (the dedupe escape-hatch: re-uploading same bytes escalates urgency/deadline, never downgrades). New version supersedes the case's prior PENDING queue_items (status='superseded', invisible to all bell/queue queries which filter explicit statuses) and auto-acks prior decided-unacknowledged items. Audit `version_uploaded` after the write block.
- Version changes summary (`_version_changes` in app.py) is a Counter multiset diff of finding labels between consecutive versions → new/resolved/still_open; it's an analysis comparison, not a text diff (UI says so). Superseded banner on non-latest contract pages takes precedence over ALL other banners. Attorney review page shows a prior_review card only when the previous version's review was actually decided (approved/rejected), not superseded-while-pending.

- Contract review page (/contracts/<id>) is a decision surface: `_contract_view` in app.py is a presentation-ONLY view builder (internal verdict enums unchanged) mapping statuses to display categories (unacceptable→Needs change, acceptable_deviation→Acceptable compromise, complies→Standard, silence→Not covered, abstain/unusual→Couldn't assess). Jargon words (abstained/silent/deviation/unacceptable) must never render; rule IDs are stripped from takeaways via `_RULE_ID_RE`; the SEC filing-header chunk (first-by-start clause matching `_PREAMBLE_RE`) is excluded from counts and shown as a muted "Document header — not analyzed" row. Headline tone must stay in lockstep with routing: `escalated=True` (pending queue item) forces amber, never green beside an attorney pill. Old review banners are absorbed into the hero status pill; there is intentionally NO "Send to attorney" button (no manual escalation endpoint exists — routing is automatic). `_citations.html` has two macros: `sources` (pills, used by review.html) and `source_line` (single muted line, contract page). Clause anchors `#clause-<id>` are revealed by JS (cards live inside collapsed <details>/groups).
- Rule-level reconciliation (post-triage): `pipeline/reconciler.py` collapses per-clause verdicts into one terminal verdict per playbook rule — met / met_via_fallback / breach / attorney_question / not_covered / outside_playbook — stored in `rule_verdicts` (DELETE+INSERT per doc in the pipeline write block). Clean rules resolve deterministically; only hedged/mixed rules get ONE audited `reconcile` LLM call each (max 6/doc); rule-less hedges bundle into a single attorney_question. "Couldn't assess" never renders in rule mode. Legacy docs and no-key mode (empty rule_verdicts) keep the per-clause view everywhere — no backfill.
- Rule-mode lockstep surfaces: `_DOCS_QUERY` risk subqueries are CASE-switched on rule rows existing (breach ≙ unacceptable; breach/fallback/question/not_covered ≙ flags; outside_playbook is FYI, never a flag); `_issue_counter` compares at rule level when rule rows exist; router escalates on breach/not_covered/attorney_question (priority 5·breach+3·gap+1·fallback+2·questions); contract.html branches on `view.rule_mode` (rule rows nest clause cards as evidence — anchor-reveal JS opens ALL ancestor <details>; clauses cited by no rule park in an "Other document clauses" details so `#clause-<id>` anchors never dangle); review.html shows a purple "Questions for you" card + reconciled positions table above the clause detail.
- Market Lens v2 (two-signal evidence): per-field rule_share (transparent frequency) + optional TabPFN conditional probability are NEVER collapsed — disagreement renders as a "worth a second look" badge. The library no longer ships an off_market boolean; the ADAPTER defines flagging as pvalue ≤ 0.05 (calibrated, from persisted omx_reference.json). The full top-k combo list must be re-derived via `score_against_reference` — `build_evidence`'s per-field rule_combo keeps only the rarest combo per field and is lossy as a list. Reports keep all v1 keys and ADD `bundle`/`synthesis`/`tabpfn_used`; contract.html renders legacy v1 rows via the no-synthesis fallback branch. Market card is role-split: operator sees the synthesized plain-language report + routing outcome only; attorney sees raw stats (Off-Market Index percentile, observed/expected/p per combo, per-field signals). Routing policy unchanged: rarity never routes — only LLM-judged unfavorable+uncovered combos escalate.

## Product

- Operator (owner role) uploads a contract with urgency (standard/high/urgent) + optional needed-by date; pipeline segments clauses, compares to the company playbook, drafts a redline .docx, and benchmarks NDAs against 1,158-contract market data (with a plain-language market position report for operators and full statistics for attorneys).
- High-risk findings auto-escalate to an attorney queue triaged by urgency → deadline → risk; attorney approves/rejects with notes; operator gets an inbox notification and acknowledges the decision.
- Urgency/deadline chips are visible on the dashboard, All Contracts, queue, contract detail, and attorney review pages.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Restart the `artifacts/syntra-app: web` workflow after editing `app.py` — Flask does not hot-reload here.
- Never call AuditLog inside an open `get_db()` write transaction (SQLite single-writer lock).
- `database.py` migrates via try/ALTER on startup; existing rows get NULL urgency, which every consumer must treat as "standard" (templates and CASE sorts already do).

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
