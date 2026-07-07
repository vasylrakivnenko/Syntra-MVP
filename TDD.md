# Syntra — Technical Design Document (v0.1)

**Scope:** Prototype for playbook-grounded contract triage. This document covers ingestion, retrieval over company knowledge, clause extraction & comparison, explainable redline generation, the attorney review queue, and the audit-trail schema.

**Design tenet (non-negotiable):** *Clear architecture, minimal code.* Every stage is a small, pure function with a typed input and a typed output. We buy the hard parts (OCR, layout parsing, embeddings, retrieval) off the shelf and write only the glue and the legal logic ourselves. A reader should be able to trace one clause from bytes to redline in a single sitting.

---

## 0. Architecture at a glance

```
                    ┌──────────── company knowledge (retrieval) ───────────┐
                    │  playbook.yaml · past contracts · approved fallbacks  │
                    └───────────────────────┬──────────────────────────────┘
                                            │ (lane two only)
 upload ──► INGEST ──► CHUNK ──► CLASSIFY ──► BRANCH ──► ROUTE ──► REDLINE ──► AUDIT
 (docx/pdf/email)   (structural) (1 LLM call)  │        (queue)   (.docx)   (append-only)
                                               ├─ Verdict   → auto-clear + redline
                                               ├─ Silence   → attorney (missing clause)
                                               └─ Abstain   → attorney (insufficient grounding)
```

**The spine is deterministic.** One cheap LLM call classifies each clause; matching a clause to its rules is a dictionary lookup, not similarity math. The only place we do vector retrieval is *lane two*, which runs solely on clauses the spine could not confidently place. This keeps the primary path fully explainable ("§7.2 → Limitation of Liability → rule `LL-1`") and cheap.

**Reused building blocks (glue only, written by us):**

| Concern | Reused component | Why |
|---|---|---|
| PDF/scanned OCR + layout, character offsets | **LandingAI ADE** (`ade-python`) | Agentic doc extraction returns structured elements with source grounding; we don't reinvent OCR/layout. |
| Native DOCX read + tracked-changes write | **python-docx** | Real Word artifact, not a web diff. |
| Email intake | **mailparse / IMAP poll** | Pull attachments, hand them to the same ingest path. |
| Retrieval over company knowledge | **ContextHub** | Drop-in hybrid retrieval layer over our corpus; used only by lane two. |
| Agentic lane-two reasoning | **smolagents** | Minimal, code-first agent loop — a few lines, no heavyweight framework. Escalate to **CrewAI** only if we need multi-role orchestration (not in prototype). |
| Classification / rationale generation | Any structured-output LLM (temp 0) | Zero-shot via playbook-as-knowledge; no fine-tune, so citations stay auditable. |

---

## 1. Ingestion (DOCX / PDF / email)

**Goal:** normalize any source into one internal `Document` model whose **character offsets are load-bearing** — every downstream citation anchors to `(start, end)` in the normalized text.

```
raw bytes ──► adapter ──► Document { docId, sourceType, elements[], fullText }
                          Element   { id, kind, text, start, end, headingPath[] }
```

- **PDF** (incl. scanned): LandingAI ADE → structured elements with grounding; we flatten to `Element[]` and compute offsets against `fullText`.
- **DOCX**: python-docx walks paragraphs/tables; native structure means offsets are exact and we keep the original file for tracked-changes export.
- **Email**: poll a mailbox, extract attachments, route each to the PDF/DOCX adapter. The email body becomes metadata (sender, received-at) on the `Document`, never a clause.

**Prototype decision (resolves open item ⚠️ §2.1 / §10.6):** ship **DOCX + PDF upload** as the live path; email intake is **built as a thin adapter but demoed optionally** — it reuses the identical ingest path, so it is code-complete without expanding the demo surface.

Each adapter is ~30–50 lines: parse → emit `Element[]` → assign offsets. One shared `Document` model, three tiny adapters.

---

## 2. Chunk — structural segmentation

No fixed-size windows. We segment by the document's own structure (article → clause → sub-clause) using the heading hierarchy ADE/python-docx already give us. Sub-clauses **inherit `headingPath`** so a nested paragraph carries its parent context into classification.

```
Element[] ──► Clause[] { id, text, start, end, headingPath[] }
```

Output is a flat list of `Clause` objects; `headingPath` preserves hierarchy without a tree. ~20 lines.

---

## 3. Classify — the spine decision

Each clause gets **one cheap LLM call**: temperature 0, structured JSON output, confidence score. The label space is the **playbook taxonomy (12–20 wedge types)** — *not* CUAD's 41. CUAD is an eval yardstick only (§6); we keep a static map from our taxonomy → CUAD where they overlap.

```json
// classify(clause) → structured output
{ "clauseType": "limitation_of_liability", "confidence": 0.91, "spans": [[412, 690]] }
```

Matching is then a **deterministic lookup**: `playbook[clauseType] → rules`. No similarity math on the primary path. This is the whole reason the system is explainable.

---

## 4. Branch — the three-way product decision

Given a classified clause and its matched rules, exactly one branch fires. This is a pure function over the classification result plus a coverage check against the playbook's required types.

| Branch | Trigger | Output payload |
|---|---|---|
| **Verdict** | Clause classified, rules matched | `complies` / `acceptable_deviation` / `unacceptable` / `unusual` + cited spans + rule IDs + one-line rationale |
| **Silence** | A playbook-**required** type has zero classified hits across the doc | "Contract is silent on X; playbook requires X" |
| **Abstain** | Low confidence, taxonomy no-fit, or lane-one/lane-two disagreement | "Insufficient grounding" + reason → attorney |

**Abstention is the anti-wrapper proof.** A wrapper always answers; Syntra visibly declines and explains why. It is surfaced prominently in the queue and the demo.

One LLM judgment call per clause with all of that clause's rules bundled (preferred / fallback / walk-away in a single prompt) — never one call per rule pair.

### 4.1 Lane two — semantic safety net (designed; stubbed in prototype)

Only runs on **low-confidence / no-fit** clauses. Uses ContextHub for hybrid **BM25 + embeddings** (exact defined terms matter) over company knowledge, then a legal-domain reranker, driven by a minimal **smolagents** loop. **Lane disagreement is itself a routing signal → abstain.** For 48h scope this lane is code-scaffolded but returns "designed, not evaluated" unless time allows; the spine ships fully.

---

## 5. Retrieval over company knowledge

The corpus = `playbook.yaml` + past reviewed contracts + attorney-approved fallbacks. It feeds two consumers:

1. **Deterministic (always):** the playbook lookup in §3 — a loaded dictionary, no retrieval needed.
2. **Semantic (lane two only):** ContextHub hybrid retrieval to find precedent for clauses the spine couldn't place.

Keeping retrieval *out* of the primary path is a deliberate architecture choice (see §9): at SMB playbook scale, deterministic matching beats vector search and stays citable.

---

## 6. Playbook schema (grounding source)

Structured data, **not prose** — YAML, versioned and diffable. Per clause type: preferred position, fallback range, walk-away, risk weight, required/optional, version. Ships with a pre-built NDA + vendor default so a customer with no legal team starts grounded.

```yaml
version: 2026-07-01
clause_types:
  limitation_of_liability:
    required: true
    risk_weight: 5           # missed high-weight clause ≫ false flag (drives eval + routing)
    rules:
      - id: LL-1
        preferred: "Cap = 12 months' fees; carve-outs for IP/confidentiality/indemnity."
        fallback:  "Cap between 12–24 months' fees."
        walk_away: "Uncapped liability, or cap below 12 months."
        cuad_map: "Cap On Liability"
```

`clauseType` from §3 keys directly into this file. Editing a rule is a one-line diff with a version bump — the entire grounding source is human-checkable.

---

## 7. Explainable redline generation (depth area)

For **Verdict** clauses that deviate but are fixable, we generate a tracked-changes `.docx` that proposes moves toward the preferred position, **each edit citing its rule ID and source span**.

```
deviating clause + rule ──► LLM proposes edited text (rationale only, grounded)
                        ──► python-docx writes tracked change + comment("rule LL-1 · §7.2")
```

- The **risk verdict** is deterministic (rule deviation → risk rubric from the playbook); the **LLM only writes the rationale/replacement prose**, never the risk decision. That split is what makes the output defensible.
- Output is a **real Word artifact with native tracked changes**, not a web-only diff — the differentiator almost nobody builds.
- No un-anchored claims: every change carries a comment linking rule ID + character offset.

---

## 8. Attorney review queue

Most contracts must **auto-clear** (green light + redline, no lawyer) — that's the economics. The queue holds only what crossed a threshold, and each item is a **pre-built brief, not a raw contract**.

Three routing signals (not just a score):

- **Whether** — all clauses comply/in-fallback and nothing required is missing → auto-clear.
- **Why** — the specific trigger (`unacceptable` / `silence` / `abstain`) with cited spans + rule attached.
- **To whom** — matter type + state + specialization (panel model for v1).

```
QueueItem {
  docId, priority(=Σ risk_weight of triggers),
  triggers: [{ branch, clauseType, ruleId, spans, reason }],
  brief: { summary, citations[], proposedRedlineDocxRef },
  assignee, status  // open · in_review · approved · rejected
}
```

**Feedback arrow (the moat):** attorney approves a deviation → optionally promoted to a new fallback rule in `playbook.yaml` (version bump) → the system routes less next time, and every escalation becomes a labeled example for a future calibrator. The attorney is a **role in the loop**, not a second designed user: a functional queue with full context.

**UPL guardrail:** framing is always "AI triages and packages; attorney advises." The threshold gate is compliance architecture, not just UX.

---

## 9. Audit trail schema

Append-only event log — the trust backbone. Every stage that produces a judgment writes one immutable event. One event view is exposed in the UI.

```
AuditEvent {
  id                 uuid
  timestamp          iso8601
  actor              "system:classify" | "system:redline" | "attorney:<id>"
  docId              string
  stage              ingest | classify | branch | redline | route | attorney_action
  model              string | null        # name+version, null for deterministic stages
  prompt_hash        sha256 | null        # reproducibility without storing raw prompts
  input_hash         sha256               # hash of the exact input (clause/doc)
  output             json                 # verdict / redline ref / decision
  citations          [{ ruleId?, docId?, start?, end? }]
  prev_hash          sha256               # chain each event to the previous → tamper-evident
}
```

- **Append-only + `prev_hash` chaining** makes the log tamper-evident without a database feature.
- Hashing prompts/inputs gives reproducibility and privilege-safety (we can prove *what* ran without storing sensitive raw text in the log).
- Privilege control is a **simple role gate** (operator vs. attorney) — present, not elaborate.

---

## 10. Evaluation (lives here, never as an in-product dashboard)

- **Public benchmarks:** CUAD (clause-extraction F1, Jaccard-threshold spans), ContractNLI (NDA entailment — maps to the wedge), ContractEval framing (correctness F1 + false-no-clause / "laziness" rate → we report our **abstention rate** against it). LegalBench-RAG only if lane two ships.
- **Named gap:** all public sets are SEC/EDGAR large-cap, well-drafted paper; SME inbound is messier — **public numbers ≠ product numbers.**
- **Bespoke golden set:** small set of real/synthetic SME NDAs + vendor contracts, scored on *weighted* recall (missed liability cap ≫ false flag), precision (no reviewer flooding), routing accuracy (human-needed vs auto-clear), and silence detection.

---

## 11. Killed alternatives (rationale kept — the panel will probe)

- **Perplexity as matcher — killed.** Measures predictability, not relevance; boilerplate scores low, dangerous bespoke drafting scores high. Salvage: perplexity vs. standard language = a *novelty detector* for v2 routing.
- **Reranker as primary matcher — demoted** to lane two. A taxonomy lookup is categorically more explainable than a 0.83 cross-encoder score.
- **Binary contradiction as output space — killed.** Floods false positives on in-fallback deviations and structurally cannot detect *absence* (pairwise checks need both paragraphs to exist; the worst risks are missing clauses).
- **Trained classifier / feature-binned NN — killed for v1.** No labels, destroys explainability, can't emit grounded redlines; binning discards the "shall vs. may" signal.
- **TabPFN / tabular foundation models — right tool, wrong stage.** Needs labeled rows at inference, which only exist after the attorney loop runs. Named **v2 routing-calibrator** candidate (small-sample tabular is its sweet spot).

---

## 12. v2 flywheel

Every attorney decision is a labeled example. Once enough accumulate, a small tabular calibrator (TabPFN-class) learns the *human-needed vs. auto-clear* boundary from the audit trail, tightening routing over time — while the deterministic playbook spine and full citation chain remain the source of truth. No fine-tuned black box ever sits between the clause and its verdict.

---

### Appendix — why the code stays small

Each stage is one typed function: `ingest → chunk → classify → branch → route → redline → audit`. Adapters and stages are 20–50 lines each; the heavy lifting (OCR, layout, embeddings, retrieval, Word I/O) is delegated to LandingAI ADE, ContextHub, python-docx, and smolagents. The only bespoke logic we own is the legal spine — classification prompt, deterministic playbook lookup, three-way branch, and the audit chain — which is exactly the part a reviewer needs to be able to read end-to-end.
