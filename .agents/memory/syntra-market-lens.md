---
name: Syntra Market Lens integration
description: Design decisions for the vendored market_lens NDA benchmarking lib and its adapter boundary.
---

# Market Lens in Syntra

Rule: `market_lens/` (vendored, at `artifacts/syntra/market_lens/`) stays byte-identical to the user's zip; `pipeline/market.py` is the ONLY file allowed to import it. Raw statistical rarity NEVER routes; the only routing path is the favorability gate (user-requested July 2026): a cheap-model assessment must judge a flagged combo unfavorable to our side AND not covered by the playbook before it escalates. The pipeline hook must swallow all Market Lens failures.

**Why:** The lib's own docs say the v1 Off-Market Index is unvalidated and saturates — real-world confirmed: both live NDAs flagged 5 combos each, and the assessment marked 5/5 unfavorable on one doc ("judge conservatively" prompt is weak), so `covered_by_playbook` dedupe is what actually suppresses noise. Keeping the lib untouched preserves upgradeability; the single-adapter seam was the user-approved plan ("modular/clean/elegant"). If clean NDAs (zero playbook findings) over-escalate in practice, add a floor (≥2 uncovered unfavorables) — only if observed.

**How to apply:** NDA perspective (mutual/recipient/discloser) is no longer statically "mutual": when the uploader confirms their party at upload (party-confirmation flow), the perspective is stored in `documents.side` for NDA docs and drives the side pill AND Market Lens favorability; `side_display`/`nda_perspective` prefer a stored perspective over the static map. Any new Market Lens feature goes through the adapter (`extract_market_row` / `score_market_row` / `run_market_lens`). Extraction rides Syntra's OpenAI client (`llm.py`) reusing the lib's hyperbolic-provider prompt — never call the lib's own providers (they need anthropic/hyperbolic keys). Scoring is fully offline against `market_data/market.sqlite` (200 ContractNLI SEC NDAs, CC BY 4.0 — keep the attribution line in the UI). Reports persist as one JSON blob in `market_reports` keyed by doc_id; the contract page renders the card only if a row exists.
