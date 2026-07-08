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
- `artifacts/syntra/market_lens/` — vendored NDA benchmarking lib (do not edit); `market_data/` holds its 200-NDA SQLite table
- `artifacts/syntra/pipeline/market.py` — the only adapter between Syntra and market_lens (extraction via llm.py, offline scoring)

## Architecture decisions

- Risk vs urgency are orthogonal: `queue_items.priority` is machine-computed risk (router weights: 5×walk-away breach + 3×missing clause + deviations + abstains, market-only escalations floored at 3); `documents.urgency`/`needed_by` are operator-declared at upload. Attorney triage sorts urgency → deadline → risk.
- Re-uploading an identical file (content-hash dedupe) is the intentional escape hatch to escalate urgency/deadline on an existing contract — it updates but never downgrades.
- AuditLog opens its own SQLite connection; audit calls must happen AFTER `with get_db()` write blocks commit, or SQLite locks ("database is locked").
- Templates share urgency/deadline chips via `templates/_chips.html` macros — edit there, not inline.
- Role-aware inbox bell is injected via `@app.context_processor` (attorney → pending queue items; operator → unacknowledged decisions on own uploads). Acknowledgment happens ONLY via the bell link (`?ack=1` on the contract page) — a plain page view/refresh must never clear the bell.
- Live updates are polling-based: base.html polls `/notifications.json` every 12s (rebuilds badge + dropdown via DOM APIs, textContent only); contract pages pending attorney review poll `/contracts/<id>/review-status` every 10s and reload when the decision lands. Endpoints must NOT live under `/api/*` — the workspace proxy routes that prefix to the separate API Server artifact.
- Grounded citations: triage snapshots the exact playbook position into `verdicts.cited_position` at analysis time (`as_of:"analysis"`) so citations survive playbook edits; rows analyzed before the feature are backfilled on read from the CURRENT playbook (`as_of:"current"`, labeled as such in the UI). Abstains are intentionally uncited.
- "Silence" verdicts mean the clause EXISTS but the playbook has no position for it (gap is in OUR playbook, not the document) — citation UIs must link the clause, not claim it's missing.
- MVP disclaimer: shared modal `#mvpModal` in base.html. Shown one-shot after uploads via `session["mvp_notice"]` popped by the `inject_mvp_notice` context processor; forced every visit on redline preview via `show_mvp_modal=True` in render_template (template context overrides processor); download links use class `mvp-guard` (JS click intercept). Navbar carries a `.mvp-sup` superscript badge.
- Citation UI is shared via `templates/_citations.html` macro (`link_prefix` arg for cross-page use); playbook matrix cells carry unconditional `pos-<sl>-<policy>` anchors, clause cards carry `clause-<id>` anchors; the .docx redline embeds plain-text SOURCES blocks (no hyperlinks).

## Product

- Operator (owner role) uploads a contract with urgency (standard/high/urgent) + optional needed-by date; pipeline segments clauses, compares to the company playbook, drafts a redline .docx, and benchmarks NDAs against 200-contract market data.
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
