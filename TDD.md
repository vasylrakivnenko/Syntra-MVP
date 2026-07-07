# Syntra — Technical Design Document (v0.4)

**Scope:** Prototype for playbook-grounded contract triage. Covers ingestion (DOCX/PDF), **playbook bootstrapping from a company's existing contracts**, retrieval over company knowledge, clause extraction & comparison, explainable redline generation, lane-two semantic safety net, attorney feedback → playbook promotion, two-role auth (owner + attorney), the attorney review queue, and the audit-trail schema backed by SQLite with a seed.json for durable persistence. Email ingestion and advanced ML calibration are explicitly scoped to later versions — see §19 Roadmap.

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
 existing contracts (bulk) ─► INGEST ─► CHUNK ─► CLASSIFY ─► AGGREGATE by type
                                                          ─► INFER positions
                                                          ─► Playbook v0.1 (draft)
                                                          ─► owner edits (manual / ask-AI) ─► save (versioned)

 B) RUNTIME — triage one inbound contract against that playbook
                    ┌──────── company knowledge (retrieval) ────────┐
                    │  playbook.yaml · past contracts · fallbacks    │
                    └───────────────────────┬────────────────────────┘
                                            │ (lane two only)
 upload ─► INGEST ─► CHUNK ─► CLASSIFY ─► BRANCH ─► ROUTE ─► REDLINE ─► AUDIT
 (docx/pdf)        (structural)(1 LLM call)  │      (queue)  (.docx)  (append-only)
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
| Playbook build | `PlaybookBuilder` | `build(contracts: list[Document]) -> Playbook` |
| Playbook edit | `PlaybookEditor` | `edit_manual(...)`, `edit_with_ai(prompt) -> Playbook`, `save() -> version` |
| Branch | `Triage` | `decide(clause, classification, playbook) -> ClauseVerdict` |
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

class PlaybookRule(BaseModel):
    id: str                   # e.g. "LL-1"
    preferred: str
    fallback: str
    walk_away: str

class PlaybookClauseType(BaseModel):
    required: bool
    risk_weight: int          # missed high-weight clause >> false flag
    rules: list[PlaybookRule]
    cuad_map: str | None = None
    source_doc_ids: list[str] = []   # provenance: which contracts this was inferred from

class Playbook(BaseModel):
    version: str
    clause_types: dict[str, PlaybookClauseType]

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

**V1 scope:** DOCX + PDF upload only. One `Document` model, two tiny adapters (~30–50 lines each). Email is pure mechanics — it just polls a mailbox and hands each attachment to the same two adapters — so it ships in v2 with no architectural change.

*Same ingest path serves both flows — bulk historical contracts (onboarding) and single inbound contracts (runtime).*

---

## 4. Chunk — structural segmentation

No fixed-size windows. Segment by the document's own structure (article → clause → sub-clause) using the heading hierarchy. Sub-clauses inherit `heading_path`, carrying parent context without a tree. Output: flat `list[Clause]`.

---

## 5. Classify — the spine decision

One cheap LLM call per clause: temperature 0, structured output, confidence. Labels come from the **playbook taxonomy (12–20 wedge types)**, not CUAD's 41 (CUAD is an eval yardstick only, §12). Matching a clause to its rules is then a **deterministic lookup** — `playbook.clause_types[clause_type].rules` — no similarity math on the primary path. That lookup is what makes the whole system explainable ("§7.2 → Limitation of Liability → rule `LL-1`").

---

## 6. Playbook Bootstrap — build v0.1 from existing contracts (new module)

**The problem owners told us:** *"We don't have a playbook — it's in our heads."* So we reconstruct it from the contracts they've already signed.

`PlaybookBuilder.build(contracts)` reuses the shared spine, then aggregates:

```
bulk contracts ─► ingest ─► chunk ─► classify        # reuse §3–5 unchanged
              ─► group clauses by clause_type
              ─► for each type: infer preferred / fallback / walk-away
                 · preferred  = the company's most common (modal) position
                 · fallback   = the observed range across their contracts
                 · walk_away  = positions they've never accepted / clear outliers
                 · risk_weight = default by type, editable
                 · source_doc_ids = provenance (grounds every inferred rule)
              ─► Playbook v0.1 (draft, versioned)
```

The LLM only **summarizes each cluster into a position statement**; the clustering and frequency are deterministic, and every inferred rule cites the `source_doc_ids` it came from — so even the bootstrapped playbook is grounded, not invented.

**Owner editing (`PlaybookEditor`) — two modes, both end in an explicit save:**
- **Manual:** edit any `preferred` / `fallback` / `walk_away` / `risk_weight` / `required` field directly (validated by the Pydantic model).
- **Ask-AI:** natural-language request (e.g. *"we never accept liability caps under 12 months"*) → a smolagents loop proposes a rule diff → owner approves → applied.
- **Save:** bumps `Playbook.version`; the change is written to the audit log (§11).

We still **ship a pre-built NDA + vendor default** so a brand-new company with zero contracts to learn from starts grounded; bootstrap refines it from their actual paper.

---

## 7. Playbook storage schema

Persisted as versioned, diffable **YAML** that round-trips 1:1 with the `Playbook` Pydantic model.

```yaml
version: 2026-07-01.1
clause_types:
  limitation_of_liability:
    required: true
    risk_weight: 5
    cuad_map: "Cap On Liability"
    source_doc_ids: ["c_0192", "c_0207", "c_0233"]   # inferred from these contracts
    rules:
      - id: LL-1
        preferred: "Cap = 12 months' fees; carve-outs for IP/confidentiality/indemnity."
        fallback:  "Cap between 12–24 months' fees."
        walk_away: "Uncapped liability, or cap below 12 months."
```

Editing a rule is a one-line diff with a version bump — the entire grounding source stays human-checkable.

---

## 8. Branch — the three-way product decision

`Triage.decide(...)` is a pure function over a classification plus a coverage check against the playbook's required types. Exactly one branch fires.

| Branch | Trigger | Output (`ClauseVerdict`) |
|---|---|---|
| **Verdict** | Classified, rules matched | `complies` / `acceptable_deviation` / `unacceptable` / `unusual` + cited spans + rule IDs + rationale |
| **Silence** | A required type has zero hits across the doc | "Contract is silent on X; playbook requires X" |
| **Abstain** | Low confidence, taxonomy no-fit, or lane disagreement | "Insufficient grounding" + reason → attorney |

**Abstention is the anti-wrapper proof** — a wrapper always answers; Syntra visibly declines and explains why. One LLM judgment call per clause with that clause's rules bundled (preferred/fallback/walk-away in a single prompt).

### 8.1 Lane two — semantic safety net (designed; stubbed in prototype)

Runs only on low-confidence / no-fit clauses. `KnowledgeIndex` (ContextHub) does hybrid **BM25 + embeddings** over company knowledge → legal reranker, driven by a minimal smolagents loop. **Lane disagreement is itself a routing signal → abstain.** Scaffolded for 48h; the deterministic spine ships fully.

---

## 9. Retrieval over company knowledge

Corpus = `playbook.yaml` + past reviewed contracts + attorney-approved fallbacks. Two consumers:
1. **Deterministic (always):** the playbook lookup in §5 — a loaded dictionary, no retrieval.
2. **Semantic (lane two only):** `KnowledgeIndex.search(...)` via ContextHub for precedent on clauses the spine couldn't place.

Keeping retrieval *off* the primary path is deliberate: at SMB playbook scale, deterministic matching beats vector search and stays citable.

---

## 10. Explainable redline generation (depth area)

For Verdict clauses that deviate but are fixable, `Redliner.redline(...)` produces a tracked-changes `.docx` proposing moves toward the preferred position, **each edit citing its rule ID and source span**.

```
deviating clause + rule ─► LLM proposes edited text (rationale only, grounded)
                       ─► python-docx writes tracked change + comment("rule LL-1 · §7.2")
```

The **risk verdict is deterministic** (rule deviation → playbook risk rubric); the LLM only writes rationale/replacement prose, never the risk decision. Output is a **real Word artifact with native tracked changes** — the differentiator almost nobody builds. No un-anchored claims.

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

**Feedback arrow (the moat):** attorney approves a deviation → optionally promoted to a new fallback rule via `PlaybookEditor` (version bump) → the system routes less next time, and every escalation becomes a labeled example for a future calibrator. The attorney is a **role in the loop**, not a second designed user.

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

- **Perplexity as matcher — killed.** Measures predictability, not relevance; boilerplate scores low, dangerous bespoke drafting scores high. Salvage: perplexity vs. standard language = a *novelty detector* for v2 routing.
- **Reranker as primary matcher — demoted** to lane two. Taxonomy lookup is categorically more explainable than a 0.83 cross-encoder score.
- **Binary contradiction as output space — killed.** Floods false positives on in-fallback deviations and cannot detect *absence* (pairwise checks need both paragraphs; the worst risks are missing clauses).
- **Trained classifier / feature-binned NN — killed for v1.** No labels, destroys explainability, can't emit grounded redlines; binning discards the "shall vs. may" signal.
- **TabPFN / tabular foundation models — right tool, wrong stage.** Needs labeled rows at inference, which only exist after the attorney loop runs. Named **v2 routing-calibrator** candidate.

---

## 15. v2 flywheel

Every attorney decision is a labeled example. Once enough accumulate in the audit trail, a small tabular calibrator (TabPFN-class) learns the *human-needed vs. auto-clear* boundary, tightening routing over time — while the deterministic playbook spine and full citation chain remain the source of truth. No fine-tuned black box ever sits between a clause and its verdict.

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
| `/playbook` — view + edit playbook (manual or ask-AI) | ✅ | ❌ |
| `/queue` — attorney review queue with pre-built briefs | ❌ | ✅ |
| `/queue/<id>/review` — approve / reject / promote to fallback | ❌ | ✅ |
| `/audit` — append-only event log view | ✅ (own docs) | ✅ (all) |

A Flask `@require_role("owner")` / `@require_role("attorney")` decorator guards each route — two simple decorators, no framework.

---

## 17. Data & Persistence (SQLite + seed.json)

**Why SQLite:** zero-config, file-based, ships with Python. Right-sized for a single-tenant prototype. No separate DB server to manage or deploy.

**Why `seed.json`:** Replit's ephemeral container means the SQLite file can be wiped on redeploy. `seed.json` is the durable store committed to the repo. The lifecycle is:

```
App startup:   seed.json ──► database.py ──► SQLite (working store)
App shutdown / export:  SQLite ──► database.py ──► seed.json (written back to disk)
```

`database.py` owns both directions — `seed_from_file()` and `dump_to_file()` — and is called from `app.py`'s startup and a `/admin/export` route.

**Tables (mirrors the Pydantic models in §2):**

| Table | Key columns | Notes |
|---|---|---|
| `documents` | `doc_id`, `source_type`, `status`, `uploaded_by`, `uploaded_at` | Metadata only; full text stored as a file ref |
| `clauses` | `clause_id`, `doc_id`, `text`, `start`, `end`, `heading_path` | Populated by `Chunker` |
| `classifications` | `clause_id`, `clause_type`, `confidence`, `spans` | Populated by `Classifier` |
| `verdicts` | `clause_id`, `branch`, `status`, `rule_ids`, `rationale` | Populated by `Triage` |
| `queue_items` | `item_id`, `doc_id`, `priority`, `assignee`, `status` | Populated by `Router` |
| `playbook_rules` | `rule_id`, `clause_type`, `preferred`, `fallback`, `walk_away`, `risk_weight`, `version` | Editable; version-bumped on every save |
| `audit_events` | `id`, `timestamp`, `actor`, `stage`, `input_hash`, `prompt_hash`, `output_json`, `prev_hash` | Append-only; `prev_hash` enforced in `AuditLog.append()` |

`seed.json` is simply a JSON dump of these tables, structured as `{ "table_name": [ ...rows ] }`. Committing it to the repo means the demo data (sample contracts, a pre-loaded playbook, a sample queue item) is always available on a fresh deploy.

---

## 18. File Structure

One directory, flat and readable. Every file has one job.

```
syntra/
├── app.py                   ← Flask app: ALL routes + Jinja2 templates (single file, ~200 lines)
│
├── models.py                ← All Pydantic models: Document, Clause, Classification,
│                              ClauseVerdict, Playbook, PlaybookRule, QueueItem,
│                              AuditEvent, Citation  (§2)
│
├── database.py              ← SQLite setup · seed_from_file() · dump_to_file()
├── seed.json                ← Canonical durable store (committed to repo)
│
├── pipeline/
│   ├── ingestor.py          ← Ingestor   — PDF (ADE) + DOCX (python-docx) → Document
│   ├── chunker.py           ← Chunker    — Document → list[Clause]
│   ├── classifier.py        ← Classifier — Clause → Classification  (1 LLM call, temp 0)
│   ├── triage.py            ← Triage     — Classification + Playbook → ClauseVerdict
│   ├── redliner.py          ← Redliner   — ClauseVerdict[] → tracked-changes .docx
│   └── router.py            ← Router     — ClauseVerdict[] → QueueItem
│
├── knowledge/
│   ├── index.py             ← KnowledgeIndex  — ContextHub-backed BM25 + embeddings
│   └── lane_two.py          ← LaneTwoAgent    — smolagents loop over KnowledgeIndex
│                              (runs only on low-confidence / no-fit clauses)
│
├── playbook/
│   ├── builder.py           ← PlaybookBuilder — infer Playbook v0.1 from existing contracts
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
| **v2** | **Email intake** | Pure mechanics: poll a mailbox, extract attachments, hand to the existing PDF/DOCX adapters. No new architecture. Deferred to keep the prototype focused, not because it is hard. |
| **v2** | Procurement / employment specialization (expanded clause taxonomy) | Wedge is NDA + vendor contracts; expansion taxonomy is a data exercise on top of the same pipeline. |
| **v3** | **TabPFN routing calibrator** | Tabular foundation model needs labeled rows at inference — those rows only exist after the attorney loop has run long enough to produce them. Named v3 so the architecture reserves the right slot (audit trail already captures the training signal). |
| **v3** | Perplexity-based novelty detector | Useful as an *attorney-routing signal* ("this clause is unusually drafted") once the primary path is battle-tested and the false-positive rate of novelty alerts can be measured. |
| **v3** | Fine-tuned legal classifier | No labeled data moat yet; the attorney flywheel in v2 generates it. Revisit when the corpus is large enough that a fine-tune would beat the zero-shot playbook prompt. |

---

### Appendix — why the code stays small *and* modular

Each stage is one class with one job (§1 module map), depending only on Pydantic models — never on another stage's internals. Adapters and stages are 20–50 lines; the heavy lifting (OCR, layout, embeddings, retrieval, Word I/O) is delegated to LandingAI ADE, ContextHub, python-docx, and smolagents. Sentrux supervises the module boundaries so the structure stays clean as the code grows. The only bespoke logic we own is the legal spine — classification prompt, deterministic playbook lookup, playbook inference, three-way branch, and the audit chain — which is exactly the part a reviewer needs to read end-to-end.
