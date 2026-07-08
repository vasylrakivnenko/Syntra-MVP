---
name: Syntra Market Lens integration
description: Design decisions for the vendored market_lens NDA benchmarking lib and its adapter boundary.
---

# Market Lens in Syntra

Rule: `market_lens/` (vendored, at `artifacts/syntra/market_lens/`) stays byte-identical to the user's zip; `pipeline/market.py` is the ONLY file allowed to import it. Market reports are advisory-only — never feed routing/triage — and the pipeline hook must swallow all Market Lens failures.

**Why:** The lib's own docs say the v1 Off-Market Index is unvalidated and saturates (most NDAs will show some flagged rare combos by construction), so it must not drive escalation. Keeping the lib untouched preserves upgradeability; the single-adapter seam was the user-approved plan ("modular/clean/elegant").

**How to apply:** Any new Market Lens feature goes through the adapter (`extract_market_row` / `score_market_row` / `run_market_lens`). Extraction rides Syntra's OpenAI client (`llm.py`) reusing the lib's hyperbolic-provider prompt — never call the lib's own providers (they need anthropic/hyperbolic keys). Scoring is fully offline against `market_data/market.sqlite` (200 ContractNLI SEC NDAs, CC BY 4.0 — keep the attribution line in the UI). Reports persist as one JSON blob in `market_reports` keyed by doc_id; the contract page renders the card only if a row exists.
