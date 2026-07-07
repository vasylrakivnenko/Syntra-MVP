# Syntra — Technical Design Document (v0.4)

**Scope:** Prototype for playbook-grounded contract triage. Covers ingestion (DOCX/PDF), **playbook bootstrapping from a company's existing contracts**, retrieval over company knowledge, clause extraction & comparison, explainable redline generation **plus a plain-language risk summary**, lane-two semantic safety net, attorney feedback → playbook promotion, two-role auth (owner + attorney), the attorney review queue, and the audit-trail schema backed by SQLite with a seed.json for durable persistence. Every redline and risk flag cites company knowledge (a playbook rule) or a source-document span — no un-anchored claims. Email ingestion and advanced ML calibration are explicitly scoped to later versions — see §19 Roadmap.

**Design tenets (non-negotiable):**
1. **Python only.** One language, end to end — pipeline, API, and UI. No JavaScript, no TypeScript, no separate frontend build step.
2. **Single-file UI.** The entire web interface lives in `app.py` (Flask). Simple routes, Jinja2 templates. A reviewer can read the whole UI in one sitting.
3. **Clear architecture, minimal code.** Every stage is a small, single-responsibility class with typed inputs and outputs. We buy the hard parts (OCR, layout, embeddings, retrieval, Word I/O) off the shelf and write only the glue and the legal logic.
4. **Modular and readable.** Named classes with explicit attributes and methods, one concern each. A reviewer can trace one clause from bytes to redline in a single sitting, and swap any stage without touching the others.
5. **Typed everywhere.** All data models are **Pydantic** models — they *are* the contract between modules, validated at every boundary.
6. **Supervised architecture.** **Sentrux** runs as the architecture-quality gate over module boundaries, so the structure stays clean as the code grows.

---

## 0. Architecture at a glance

Two flows share the same spine (`ingest → chunk → classify`):

```
 A) ONBOARDING — build the playbook the owner never wrote down
 existing contracts (bulk) ─► INGEST ─► CHUNK ─► CLASSIFY
                          ─► CLUSTER by (side, service line)          # matrix ROWS
                          ─► for each row × policy: INFER position     # matrix CELLS
                          ─► Playbook v0.1 = service × policy matrix (draft)
                          ─► owner edits cells · renames/adds/deletes rows & cols ─► save (versioned)

 B) RUNTIME — triage one inbound contract against that playbook
                    ┌──────── company knowledge (retrieval) ────────┐
                    │  playbook.yaml · past contracts · fallbacks    │
                    └───────────────────────┬────────────────────────┘
                                            │ (lane two only)
 upload ─► INGEST ─► SEGMENT ─► CHUNK ─► CLASSIFY ─► BRANCH ─► ROUTE ─► REDLINE + SUMMARY ─► AUDIT
 (docx/pdf)        (side+service)(struct.) (1 call)   │      (queue)  (.docx + plain-language) (append-only)
                                             ├─ Verdict  → auto-clear + redline
                                             ├─ Silence  → attorney (missing clause)
                                             └─ Abstain  → attorney (insufficient grounding)
```

**Why this matters (validated with owners):** SMB owners don't have a written playbook — *their standards live in their heads and in the contracts they've already signed.* So we don't ask them to author one. We **derive playbook v0.1 from their existing contracts**, then let them correct it. The runtime spine stays deterministic; the playbook is just data that flow A produces and flow B consumes.

---

## 1. Stack & reused building blocks

We write glue and legal logic; everything hard is delegated.

| Concern | Component | Why |
|---|---|---|
| **Language** | **Python (only)** | One language end to end — pipeline, UI, and tooling. No JS/TS build step. |
| **UI** | **Flask + Jinja2** (single file: `app.py`) | Minimal, readable. Routes + templates in one place; no frontend framework. |
| **Auth / SSO** | **Replit Auth** (OpenID Connect + PKCE) | Native Replit sign-in — no password management; role is selected at first login and stored in session. |
| **Database** | **SQLite** (via `sqlite3`) | Zero-config, file-based, ships with Python. Enough for a single-tenant prototype. |
| **Seed / persistence** | `seed.json` ↔ `database.py` | Canonical data store that survives publish/unpublish cycles. On startup: load `seed.json` → seed DB. On shutdown / export: dump DB → `seed.json`. |
| Data models & validation | **Pydantic** | One typed contract per module boundary; validation is free. |
| Architecture-quality supervision | **Sentrux** | Structural/fitness checks over module boundaries — keeps the layering honest as code grows. |
| PDF/scanned OCR + layout + offsets | **LandingAI ADE** (`ade-python`) | Agentic extraction with source grounding; no reinventing OCR. |
| Native DOCX read + tracked-changes write | **python-docx** | Real Word artifact, not a web diff. |
| Email intake *(v2)* | **mailparse / IMAP poll** | Pure mechanics — pulls attachments into the same ingest path; no new architecture needed. |
| Retrieval over company knowledge | **ContextHub** (`github.com/andrewyng/context-hub`) | Drop-in hybrid retrieval over our corpus; used only by lane two. |
| Agentic lane-two + ask-AI editing | **smolagents** | Minimal, code-first agent loop. Escalate to **CrewAI** only if multi-role orchestration is ever needed (not in prototype). |
| Classification / rationale / inference | Structured-output LLM (temp 0) | Zero-shot via playbook-as-knowledge; no fine-tune, so citations stay auditable. |

**Module map (each is one class, one responsibility):**

| Module | Class | Key methods |
|---|---|---|
| Ingest | `Ingestor` | `ingest(bytes, source_type) -> Document` |
| Chunk | `Chunker` | `chunk(doc: Document) -> list[Clause]` |
| Classify | `Classifier` | `classify(clause: Clause) -> Classification` |
| Segment | `Segmenter` | `segment(doc: Document) -> Segment` (doc-level: side + service line) |
| Playbook build | `PlaybookBuilder` | `build(contracts: list[Document]) -> Playbook` |
| Playbook edit | `PlaybookEditor` | `edit_manual(...)`, `edit_with_ai(prompt) -> Playbook`, `save() -> version` |
| Branch | `Triage` | `decide(clause, classification, segment, playbook) -> ClauseVerdict` |
| Retrieval | `KnowledgeIndex` | `search(query) -> list[Passage]` (ContextHub-backed) |
| Redline | `Redliner` | `redline(doc, verdicts) -> DocxRef` |
| Route | `Router` | `route(verdicts) -> QueueItem` |
| Audit | `AuditLog` | `append(event: AuditEvent) -> AuditEvent` |

Stages depend only on Pydantic models, never on each other's internals — that is the seam Sentrux enforces.

---

## 2. Core data models (Pydantic)

These models are the whole interface surface. Everything below is expressed in terms of them.

```python
from pydantic import BaseModel

class Element(BaseModel):
    id: str
    kind: str                 # heading | paragraph | table_cell ...
    text: str
    start: int; end: int      # char offsets into Document.full_text (load-bearing)
    heading_path: list[str]   # article > clause > sub-clause

class Document(BaseModel):
    doc_id: str
    source_type: str          # docx | pdf | email
    full_text: str
    elements: list[Element]

class Clause(BaseModel):
    id: str; text: str
    start: int; end: int
    heading_path: list[str]

class Classification(BaseModel):
    clause_type: str          # from the playbook taxonomy, NOT CUAD
    confidence: float
    spans: list[tuple[int, int]]

class Segment(BaseModel):            # which matrix ROW an inbound contract belongs to
    side: str                        # supplier (we buy) | customer (we sell)
    service_line: str                # matches a ServiceLine.id, e.g. "fuel_cards"
    confidence: float

class Position(BaseModel):           # one matrix CELL = one citable rule
    id: str                          # stable ID for citations, e.g. "PD-FUEL-1"
    preferred: str
    fallback: str
    walk_away: str
    risk_weight: int | None = None   # overrides the column default when set
    source_doc_ids: list[str] = []   # provenance: contracts this cell was inferred from

class PolicyColumn(BaseModel):       # a COLUMN of the matrix
    id: str                          # "payment_deferral"
    label: str
    clause_type: str                 # links to the runtime clause taxonomy (§5)
    required: bool = False
    risk_weight: int = 3             # column default; a cell may override
    cuad_map: str | None = None

class ServiceLine(BaseModel):        # a ROW of the matrix
    id: str                          # "fuel_cards"
    label: str
    side: str                        # supplier | customer
    positions: dict[str, Position]   # keyed by PolicyColumn.id; missing key = gap

class Playbook(BaseModel):
    version: str
    policies: list[PolicyColumn]     # columns, shared across all rows
    service_lines: list[ServiceLine] # rows

class ClauseVerdict(BaseModel):
    clause_id: str
    branch: str               # verdict | silence | abstain
    status: str | None        # complies | acceptable_deviation | unacceptable | unusual
    rule_ids: list[str]
    spans: list[tuple[int, int]]
    rationale: str
    reason: str | None = None # for abstain/silence
```

---

## 3. Ingestion (DOCX / PDF)

Normalize any source into one `Document`; **character offsets are load-bearing** — every citation anchors to `(start, end)`.

- **PDF** (incl. scanned): LandingAI ADE → structured elements with grounding → flatten to `Element[]`.
- **DOCX**: python-docx walks paragraphs/tables; offsets are exact and we keep the original for tracked-changes export.

**V1 scope:** DOCX + PDF **contract** upload only. One `Document` model, two tiny adapters (~30–50 lines each). Email is pure mechanics — it just polls a mailbox and hands each attachment to the same two adapters — so it ships in v2 with no architectural change. The challenge's other intake type, free-form **legal requests** (a question rather than a document), rides the same branch → route → audit machinery once ingested and is a v2 channel; the contract wedge is where grounded redlining proves the thesis.

*Same ingest path serves both flows — bulk historical contracts (onboarding) and single inbound contracts (runtime).*

---

## 4. Chunk — structural segmentation

No fixed-size windows. Segment by the document's own structure (article → clause → sub-clause) using the heading hierarchy. Sub-clauses inherit `heading_path`, carrying parent context without a tree. Output: flat `list[Clause]`.

---

## 5. Classify — the spine decision

One cheap LLM call per clause: temperature 0, structured output, confidence. Labels come from the **playbook taxonomy (12–20 wedge types)**, not CUAD's 41 (CUAD is an eval yardstick only, §13). Matching is then a **two-axis deterministic lookup**: the document's **segment** (side + service line, §5.1) selects the matrix **row**, the clause's `clause_type` selects the **column**, and the cell is the applicable position — `playbook[segment][clause_type]`. No similarity math on the primary path. That lookup is what makes the whole system explainable ("§7.2 → Fuel cards ▸ Payment deferral → rule `PD-FUEL-1`").

**The wedge taxonomy (12–20 types, playbook-defined):** limitation of liability · indemnification · term & termination · confidentiality scope · IP ownership · payment terms · governing law & jurisdiction · auto-renewal · data protection / DPA · assignment & change of control · warranties & disclaimers · dispute resolution · insurance · non-solicitation. These are the exact labels the classifier emits and the **policy columns** of the playbook matrix (§6); CUAD's 41 categories are mapped in only where they overlap, purely for benchmark reporting (§13).

### 5.1 Segmentation — pick the matrix row

Because the playbook is a matrix (§6), triage first has to know **which row** an inbound contract belongs to. `Segmenter.segment(doc)` makes one document-level LLM call returning a `Segment`: the **side** (are we buyer or seller?) and the **service line** (fuel cards, tyres, vehicle lease, freight…), with a confidence. The segment selects the matrix **row**; per-clause classification then selects the **column** — together they pick the cell.

If the segment is low-confidence, or the service line isn't in the matrix yet, triage **abstains** and routes to the attorney, surfacing the unknown service line as a **candidate new row** for the owner to confirm. A new service line never receives a silent, ungrounded verdict.

---

## 6. Playbook Bootstrap — build the service × policy matrix from existing contracts

**The problem owners told us:** *"We don't have a playbook — it's in our heads."* And their standards are not one-size-fits-all: a trucking company runs **15-day terms on fuel but 60-day terms on a vehicle lease**, and its *supplier* paper looks nothing like its *customer* paper. A single flat list of positions blurs all of that together. So the playbook is a **matrix** — rows = service lines (each tagged supplier or customer), columns = policies, every cell = the position that actually applies:

| Service line ▸ Policy | Payment deferral | Liability cap | … |
|---|---|---|---|
| **Fuel cards** *(supplier)* | 30 days EOM | 12 mo fees | … |
| **Tyres** *(supplier)* | 60 days from invoice | … | … |
| **Freight** *(customer)* | 15 days EOM | … | … |

`PlaybookBuilder.build(contracts)` reuses the shared spine, then fills the matrix in two passes:

```
bulk contracts ─► ingest ─► chunk ─► classify              # reuse §3–5 unchanged
   PASS 1 ─► CLUSTER each contract into a (side, service_line) row
             · side         = are we buyer or seller?  (parties + who owes what)
             · service_line = which product/service?   (subject, titles, defined terms)
   PASS 2 ─► for each row × policy column, INFER the cell
             · preferred   = the modal position in that cluster
             · fallback    = the observed range across those contracts
             · walk_away   = positions never accepted / clear outliers
             · source_doc_ids = provenance for that specific cell
          ─► Playbook v0.1 = a filled matrix (draft, versioned)
```

The LLM does only summarization — naming each cluster and turning a cell's observed positions into a statement — while the **clustering counts and frequencies stay deterministic**, and **every cell cites the `source_doc_ids` it came from.** A service line with no observed position for a policy is left **blank, not guessed**.

**Owner editing (`PlaybookEditor`) — the whole matrix is editable; every change ends in an explicit save:**
- **Cells:** edit any `preferred` / `fallback` / `walk_away` / `risk_weight` / `required`.
- **Rows & columns:** rename, add, or delete service lines (rows) and policies (columns) — e.g. split "fuel" into "fuel cards" and "bulk diesel," or add a "data protection" column.
- **Ask-AI:** natural-language request (e.g. *"we never accept liability caps under 12 months on customer freight"*) → a smolagents loop proposes a matrix diff → owner approves → applied.
- **Save:** bumps `Playbook.version`; the change is written to the audit log (§12).

We still **ship a pre-built NDA + vendor default** (a small starter matrix) so a brand-new company with zero contracts starts grounded; bootstrap refines it from their actual paper.

---

## 7. Playbook storage schema — the matrix as YAML

Persisted as versioned, diffable **YAML** that round-trips 1:1 with the `Playbook` Pydantic model: a list of policy **columns** and a list of service-line **rows**, each row holding one position per policy.

```yaml
version: 2026-07-01.1

policies:                                   # COLUMNS — shared across every row
  - id: payment_deferral
    label: "Payment deferral"
    clause_type: payment_terms
    required: true
    risk_weight: 4
    cuad_map: "Payment Terms"
  - id: limitation_of_liability
    label: "Limitation of liability"
    clause_type: limitation_of_liability
    required: true
    risk_weight: 5
    cuad_map: "Cap On Liability"

service_lines:                              # ROWS — each tagged with the side we're on
  - id: fuel_cards
    label: "Fuel cards"
    side: supplier                          # we BUY fuel cards from a supplier
    positions:
      payment_deferral:
        id: PD-FUEL-1
        preferred: "30 days EOM"
        fallback:  "15–45 days EOM"
        walk_away: "Payment on receipt / prepayment"
        source_doc_ids: ["c_0192", "c_0207"]
  - id: tyres
    label: "Tyres"
    side: supplier
    positions:
      payment_deferral:
        id: PD-TYRE-1
        preferred: "60 days from invoice received"
        fallback:  "45–75 days from invoice"
        walk_away: "Under 30 days"
        source_doc_ids: ["c_0233"]
  - id: freight_services
    label: "Freight services"
    side: customer                          # we SELL freight to our customers
    positions:
      payment_deferral:
        id: PD-FRT-1
        preferred: "15 days EOM"
        fallback:  "30 days EOM"
        walk_away: "Over 45 days"
        source_doc_ids: ["c_0301", "c_0305"]
```

Editing a cell — or renaming, adding, or deleting a row or column — is a small, reviewable YAML diff with a version bump, so the entire grounding source stays human-checkable. A blank cell (no position for that service × policy) is an explicit gap, not a guess.

---

## 8. Branch — the three-way product decision

`Triage.decide(...)` is a pure function over a classification, the document's **segment** (which selects the matrix row, §5.1), and a coverage check against that row's required policies. Exactly one branch fires.

| Branch | Trigger | Output (`ClauseVerdict`) |
|---|---|---|
| **Verdict** | Classified, matrix cell matched | `complies` / `acceptable_deviation` / `unacceptable` / `unusual` + cited spans + rule IDs + rationale |
| **Silence** | A policy the matrix marks required *for this row* has zero clause hits | "Contract is silent on X; the playbook requires X for this service line" |
| **Abstain** | Low confidence, taxonomy no-fit, **unknown segment** (new service line not in the matrix), or lane disagreement | "Insufficient grounding" + reason → attorney; unknown service lines are flagged as candidate new rows |

**Abstention is the anti-wrapper proof** — a wrapper always answers; Syntra visibly declines and explains why. One LLM judgment call per clause with the matched cell's position bundled (preferred/fallback/walk-away in a single prompt).

### 8.1 Lane two — semantic safety net (live in v1)

Runs only on low-confidence / no-fit clauses — never on the primary path. `KnowledgeIndex` (ContextHub) does hybrid **BM25 + embeddings** over company knowledge → legal reranker, driven by a minimal smolagents loop. **Lane disagreement is itself a routing signal → abstain.** The deterministic spine still decides every clause it can place; lane two only catches what the spine couldn't, and its job is to *escalate*, never to override a deterministic verdict. It ships live in v1.

---

## 9. Retrieval over company knowledge

Corpus = `playbook.yaml` + past reviewed contracts + attorney-approved fallbacks. Two consumers:
1. **Deterministic (always):** the playbook lookup in §5 — a loaded dictionary, no retrieval.
2. **Semantic (lane two only):** `KnowledgeIndex.search(...)` via ContextHub for precedent on clauses the spine couldn't place.

Keeping retrieval *off* the primary path is deliberate: at SMB playbook scale, deterministic matching beats vector search and stays citable.

---

## 10. Explainable redline + plain-language risk summary (depth area)

For Verdict clauses that deviate but are fixable, `Redliner.redline(...)` produces a tracked-changes `.docx` proposing moves toward the preferred position, **each edit citing its rule ID and source span**.

```
deviating clause + rule ─► LLM proposes edited text (rationale only, grounded)
                       ─► python-docx writes tracked change + comment("rule LL-1 · §7.2")
```

The **risk verdict is deterministic** (rule deviation → playbook risk rubric); the LLM only writes rationale/replacement prose, never the risk decision. Output is a **real Word artifact with native tracked changes** — the differentiator almost nobody builds. No un-anchored claims.

### 10.1 Plain-language risk summary

Alongside the `.docx`, every triaged contract produces a **plain-language risk summary** for the owner — the challenge's second required output, next to the redline. It is assembled deterministically from the `ClauseVerdict[]`, not free-written by the model:

- **One line per flagged clause:** what the clause says, what the playbook wants, and the gap — in plain English, no legalese.
- **Every line cites its source:** the playbook `rule_id` and/or the character-offset span in the source document (§2 `Citation`). No un-anchored claims.
- **Ordered by risk:** clauses sort by `risk_weight`, so the owner reads the dangerous items first.
- **Overall disposition:** auto-cleared · needs attorney — with the count and the specific reason (`unacceptable` / `silence` / `abstain`) for each escalation.

The summary is a *view over the same `ClauseVerdict[]`* that drives the redline and the audit trail, so the three can never disagree.

---

## 11. Attorney review queue

Most contracts must **auto-clear** (green light + redline, no lawyer) — that's the economics. The queue holds only threshold-crossing items, each a **pre-built brief, not a raw contract**. Three routing signals:

- **Whether** — all clauses comply/in-fallback and nothing required missing → auto-clear.
- **Why** — the specific trigger (`unacceptable` / `silence` / `abstain`) with cited spans + rule.
- **To whom** — matter type + state + specialization (panel model for v1).

```python
class QueueItem(BaseModel):
    doc_id: str
    priority: int                      # Σ risk_weight of triggers
    triggers: list[ClauseVerdict]
    proposed_redline_ref: str | None
    assignee: str | None = None
    status: str = "open"               # open | in_review | approved | rejected
```

**Feedback arrow (the moat) — live in v1:** when an attorney approves a deviation, the review screen offers a one-click **"promote to fallback"** action → `PlaybookEditor` writes the new fallback rule, bumps `Playbook.version`, and logs the edit to the audit trail (§12) → the system routes that pattern less next time. Promotion is **attorney-approved, never silent** — the human decision is what creates the rule — and every escalation also becomes a labeled example for the future routing calibrator (§15). The attorney is a **role in the loop**, not a second designed user.

**UPL guardrail:** framing is always "AI triages and packages; attorney advises." The threshold gate is compliance architecture, not just UX.

---

## 12. Audit trail schema

Append-only event log — the trust backbone. Every judgment-producing stage (including playbook edits) writes one immutable event; one trace view is exposed in the UI.

```python
class Citation(BaseModel):
    rule_id: str | None = None
    doc_id: str | None = None
    start: int | None = None
    end: int | None = None

class AuditEvent(BaseModel):
    id: str
    timestamp: str
    actor: str                 # system:classify | system:redline | attorney:<id> | owner:<id>
    doc_id: str | None
    stage: str                 # ingest|classify|branch|redline|route|playbook_edit|attorney_action
    model: str | None          # name+version, null for deterministic stages
    prompt_hash: str | None    # reproducibility without storing raw prompts
    input_hash: str            # hash of the exact input
    output: dict               # verdict / redline ref / decision / playbook diff
    citations: list[Citation]
    prev_hash: str             # chains each event to the previous → tamper-evident
```

- **Append-only + `prev_hash` chaining** → tamper-evident without a special DB feature.
- Hashing prompts/inputs → reproducibility and privilege-safety (prove *what* ran without storing sensitive raw text).
- Privilege control is a **simple role gate** (owner/operator vs. attorney) — present, not elaborate.

---

## 13. Evaluation (lives here, never as an in-product dashboard)

- **Public benchmarks:** CUAD (clause-extraction F1, Jaccard spans), ContractNLI (NDA entailment → wedge), ContractEval framing (correctness F1 + false-no-clause rate → we report our **abstention rate**). LegalBench-RAG only if lane two ships.
- **Named gap:** public sets are SEC/EDGAR large-cap, well-drafted; SME inbound is messier — **public numbers ≠ product numbers.**
- **Bespoke golden set:** small real/synthetic SME NDAs + vendor contracts, scored on *weighted* recall (missed liability cap ≫ false flag), precision (no reviewer flooding), routing accuracy, silence detection.
- **Bootstrap-specific check:** does an inferred playbook v0.1 agree with an attorney's hand-authored playbook on the same contract set? (measures onboarding quality).

---

## 14. Killed alternatives (rationale kept — the panel will probe)

- **Perplexity as matcher — killed.** Measures predictability, not relevance; boilerplate scores low, dangerous bespoke drafting scores high. Salvage: perplexity vs. standard language = a *novelty detector* for v3 routing.
- **Reranker as primary matcher — demoted** to lane two. Taxonomy lookup is categorically more explainable than a 0.83 cross-encoder score.
- **Binary contradiction as output space — killed.** Floods false positives on in-fallback deviations and cannot detect *absence* (pairwise checks need both paragraphs; the worst risks are missing clauses).
- **Trained classifier / feature-binned NN — killed for v1.** No labels, destroys explainability, can't emit grounded redlines; binning discards the "shall vs. may" signal.
- **TabPFN / tabular foundation models — right tool, wrong stage.** Needs labeled rows at inference, which only exist after the attorney loop runs. Named **v3 routing-calibrator** candidate (§15).

---

## 15. Attorney flywheel & future calibrator

The flywheel has two stages on different timelines:

- **Feedback capture (live in v1):** every attorney decision — approve, reject, or promote-to-fallback (§11) — is written to the audit trail as a labeled example. This is on from day one; it is what makes the playbook improve and what accumulates the training signal.
- **Routing calibrator (v3):** once enough labeled decisions accumulate, a small tabular calibrator (TabPFN-class, §19) learns the *human-needed vs. auto-clear* boundary and tightens routing over time.

Throughout, the deterministic playbook spine and full citation chain remain the source of truth. **No fine-tuned black box ever sits between a clause and its verdict** — the calibrator only tunes the routing threshold, never the legal decision.

---

## 16. Auth & Access Control

**Mechanism:** Replit Auth (OpenID Connect + PKCE). No password management; the user clicks "Sign in with Replit" and is authenticated by the platform.

**Two roles, one prompt:** This is a single-tenant prototype with exactly **one owner account and one attorney account** — both pre-seeded in `seed.json`. After Replit Auth confirms identity, the app presents a single role-selection screen:

```
┌─────────────────────────────────────┐
│  Welcome to Syntra.                 │
│  Who are you signing in as?         │
│                                     │
│   [ Owner / Operator ]              │
│   [ Supervising Attorney ]          │
└─────────────────────────────────────┘
```

Role is stored in the Flask session. No user creation flow exists in the prototype — if a third Replit identity tries to log in, they see "Access not configured." That's intentional; the demo has two seats.

**What each role sees:**

| Route | Owner | Attorney |
|---|---|---|
| `/upload` — upload a contract | ✅ | ❌ |
| `/contracts` — list of uploaded contracts + status | ✅ | ❌ |
| `/contracts/<id>` — triage result, redline, risk summary | ✅ (read) | ❌ |
| `/playbook` — view + edit the service × policy matrix (cells, rows, columns; manual or ask-AI) | ✅ | ❌ |
| `/queue` — attorney review queue with pre-built briefs | ❌ | ✅ |
| `/queue/<id>/review` — approve / reject / promote to fallback | ❌ | ✅ |
| `/audit` — append-only event log view | ✅ (own docs) | ✅ (all) |

A Flask `@require_role("owner")` / `@require_role("attorney")` decorator guards each route — two simple decorators, no framework.

---

## 17. Data & Persistence — three layers

Storage is split by the nature of each artifact, not by convenience. Each layer has exactly one home.

**Layer 1 — Playbook → YAML in git (`playbook/default.yaml`)**
The playbook is the grounding source of truth. It lives as a YAML file committed to the repo: versioning is git history, diffs are human-readable, and any attorney or operator can edit it in a text editor. Putting it in a database would destroy its best property — traceability. When `PlaybookEditor` saves a change, it writes a new YAML file and bumps the `version` field; the old version is preserved in git history automatically.

**Layer 2 — Contracts + derived artifacts → SQLite tables + files on disk**
Everything the pipeline produces from an uploaded contract is stored in SQLite and the local filesystem. The SQLite file is the working store; `seed.json` (committed to the repo) is its durable twin so a fresh deploy starts with real demo data.

```
App startup:          seed.json ──► database.py ──► SQLite
App shutdown/export:  SQLite    ──► database.py ──► seed.json  (via /admin/export)
```

| Table | Key columns | Populated by |
|---|---|---|
| `documents` | `doc_id`, `source_type`, `status`, `side`, `service_line`, `uploaded_by`, `uploaded_at` | `Ingestor` + `Segmenter` |
| `clauses` | `clause_id`, `doc_id`, `text`, `start`, `end`, `heading_path` | `Chunker` |
| `classifications` | `clause_id`, `clause_type`, `confidence`, `spans` | `Classifier` |
| `verdicts` | `clause_id`, `branch`, `status`, `rule_ids`, `rationale` | `Triage` |
| `queue_items` | `item_id`, `doc_id`, `priority`, `assignee`, `status` | `Router` |

Parsed document text and generated `.docx` redlines are stored as files on disk; the table rows hold a `file_ref` path, not the raw bytes.

**Layer 3 — Audit trail → single append-only SQLite table**
The `audit_events` table is insert-only. There are **no `UPDATE` or `DELETE` statements anywhere in the codebase** — that discipline is enforced by `AuditLog.append()` being the only write path, and it is demonstrable in code review, which is exactly the kind of build evidence the panel verifies.

| Column | Purpose |
|---|---|
| `id` | UUID |
| `timestamp` | ISO 8601 |
| `actor` | `system:classify` · `system:redline` · `owner:<id>` · `attorney:<id>` |
| `stage` | `ingest` · `classify` · `branch` · `redline` · `route` · `playbook_edit` · `attorney_action` |
| `model` | Model name + version, or `null` for deterministic stages |
| `input_hash` | SHA-256 of the exact input — proves what was processed without storing sensitive text |
| `prompt_hash` | SHA-256 of the prompt — reproducibility without storing raw prompts |
| `output_json` | The verdict, diff, or decision |
| `citations` | JSON array of `{ rule_id?, doc_id?, start?, end? }` |
| `prev_hash` | SHA-256 of the previous event — chains the log, making tampering detectable |

---

## 18. File Structure

One directory, flat and readable. Every file has one job.

```
syntra/
├── app.py                   ← Flask app: ALL routes + Jinja2 templates (single file, ~200 lines)
│
├── models.py                ← All Pydantic models: Document, Clause, Classification,
│                              Segment, ClauseVerdict, Playbook (PolicyColumn ·
│                              ServiceLine · Position), QueueItem, AuditEvent, Citation  (§2)
│
├── database.py              ← SQLite setup · seed_from_file() · dump_to_file()
├── seed.json                ← Canonical durable store (committed to repo)
│
├── pipeline/
│   ├── ingestor.py          ← Ingestor   — PDF (ADE) + DOCX (python-docx) → Document
│   ├── segmenter.py         ← Segmenter  — Document → Segment  (doc-level: side + service line)
│   ├── chunker.py           ← Chunker    — Document → list[Clause]
│   ├── classifier.py        ← Classifier — Clause → Classification  (1 LLM call, temp 0)
│   ├── triage.py            ← Triage     — Classification + Segment + Playbook → ClauseVerdict
│   ├── redliner.py          ← Redliner   — ClauseVerdict[] → tracked-changes .docx
│   └── router.py            ← Router     — ClauseVerdict[] → QueueItem
│
├── knowledge/
│   ├── index.py             ← KnowledgeIndex  — ContextHub-backed BM25 + embeddings
│   └── lane_two.py          ← LaneTwoAgent    — smolagents loop over KnowledgeIndex
│                              (runs only on low-confidence / no-fit clauses)
│
├── playbook/
│   ├── builder.py           ← PlaybookBuilder — cluster contracts → infer service × policy matrix
│   ├── editor.py            ← PlaybookEditor  — manual edit + ask-AI (smolagents) + save
│   └── default.yaml         ← Pre-built NDA + vendor playbook (used if no contracts exist yet)
│
└── audit.py                 ← AuditLog — append(event) with prev_hash chaining
```

**Rule:** `app.py` imports from `pipeline/`, `knowledge/`, `playbook/`, and `audit.py` — but those modules **never import from each other** or from `app.py`. Data flows through Pydantic models only. Sentrux enforces this boundary.

---

## 19. Roadmap

Decisions about what is *not* in the prototype are as deliberate as what is. The table below states the version, the feature, and the reason it is not v1 — so the panel can probe the reasoning, not just the exclusion.

| Version | Feature | Why deferred |
|---|---|---|
| **v1 (prototype)** | DOCX + PDF upload · playbook bootstrap · three-way branch · deterministic redline · lane-two semantic safety net · attorney feedback → playbook rule promotion · two-role auth (owner + attorney) · attorney queue · audit trail · SQLite + seed.json | The full core loop — everything needed to prove the thesis and run a real demo. |
| **v2** | **Email + legal-request intake** | Email is pure mechanics: poll a mailbox, extract attachments, hand to the existing PDF/DOCX adapters. Free-form legal requests (a question rather than a document) ride the same branch → route → audit path once ingested. No new architecture — deferred to keep the prototype focused, not because it is hard. |
| **v2** | Procurement / employment specialization (expanded clause taxonomy) | Wedge is NDA + vendor contracts; expansion taxonomy is a data exercise on top of the same pipeline. |
| **v3** | **TabPFN routing calibrator** | Tabular foundation model needs labeled rows at inference — those rows only exist after the attorney loop has run long enough to produce them. Named v3 so the architecture reserves the right slot (audit trail already captures the training signal). |
| **v3** | Perplexity-based novelty detector | Useful as an *attorney-routing signal* ("this clause is unusually drafted") once the primary path is battle-tested and the false-positive rate of novelty alerts can be measured. |
| **v3** | Fine-tuned legal classifier | No labeled data moat yet; the attorney flywheel (feedback capture live in v1) generates it. Revisit when the corpus is large enough that a fine-tune would beat the zero-shot playbook prompt. |

---

## 20. Architecture Q&A (anticipated panel probes)

The decisions above imply answers to the questions a technical panel will ask. Stated plainly so the reasoning is on record:

| Probe | Answer |
|---|---|
| Why structured lookup over a vector DB? | At SMB playbook scale, deterministic clause-type → rule matching beats semantic search and stays citable. Embeddings are reserved for lane two (precedent) where there is no exact rule to match. |
| Why prompt + grounding over fine-tuning? | No labeled-data moat yet, and auditability demands citations, not weights. A fine-tune would trade explainability for accuracy we can't yet measure. |
| What about 10× latency? | Contract review is minutes-scale and asynchronous, not chat. Slower throughput is a **queue problem, not an architecture crisis** — the attorney queue already absorbs it. |
| What happens when the model is wrong? | Three overlapping guards: the abstention branch (§8), the attorney threshold gate (§11), and the append-only audit trail (§12). The risk verdict itself is deterministic (playbook-deviation rubric); the LLM only writes rationale, never the decision. |
| Why an LLM at all, over trained ML? | Zero-shot via playbook-as-knowledge, generative redlines are required, and there are no labels until the flywheel runs (§14, §15). |
| How is this not a wrapper? | A wrapper always answers. Syntra visibly **declines and explains** (abstain/silence), grounds every output in a cited rule or span, and emits a native Word artifact — none of which a thin prompt wrapper does. |

---

### Appendix — why the code stays small *and* modular

Each stage is one class with one job (§1 module map), depending only on Pydantic models — never on another stage's internals. Adapters and stages are 20–50 lines; the heavy lifting (OCR, layout, embeddings, retrieval, Word I/O) is delegated to LandingAI ADE, ContextHub, python-docx, and smolagents. Sentrux supervises the module boundaries so the structure stays clean as the code grows. The only bespoke logic we own is the legal spine — classification prompt, deterministic playbook lookup, playbook inference, three-way branch, and the audit chain — which is exactly the part a reviewer needs to read end-to-end.
