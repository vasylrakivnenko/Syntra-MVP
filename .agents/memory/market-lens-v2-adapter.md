---
name: Market Lens v2 adapter pitfalls
description: Non-obvious contracts when adapting market_lens v2 (evidence bundle, flagging, combo list)
---

- **Lossy per-field combo view:** `build_evidence()`'s `fields[*].rule_combo` keeps only
  the RAREST combo per field (`setdefault`). Reconstructing "all contributions" from it
  silently drops distinct combos whose fields are all claimed by rarer ones — and flagged
  combos feed favorability → attorney routing. The list of record must come from
  `score_against_reference()` directly.
  **Why:** discovered in review — a 5-combo doc stored only 4.
- **Flagging is adapter policy, not library output:** v1.1 contributions are
  (combo, observed, expected, pvalue) tuples with no off_market boolean. Syntra defines
  flagged = pvalue ≤ 0.05; contributions without a p-value (legacy path, i.e.
  `market_data/omx_reference.json` missing) are never flagged — that path also logs a
  loud warning because it silently disables market escalation.
- **Two signals never collapse:** rule_share (marginal frequency) and TabPFN p_obs
  (conditional probability) are different quantities; UI shows both + a disagreement
  badge (both present, exactly one < 0.15). Synthesis prompt explicitly forbids picking
  a winner.
- **TabPFN cost contract:** `fit_all_fields` is ~18 network calls — once per process,
  behind a lock, failures cached; inert without TABPFN_TOKEN.
