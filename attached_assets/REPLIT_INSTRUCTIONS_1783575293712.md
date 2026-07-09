# Quick start for the Replit agent

Read this first. Full detail (evidence bundle schema, TabPFN cost model,
lawyer-vs-client view guidance) is in **`spec.md`** — read that before
building anything, this file is just the fast path to get it running.

## 1. Unzip and install

```bash
unzip legalens_market_lens_v2.zip
cd legalens_market_lens
pip install -r requirements.txt
```

Python 3.10+.

## 2. Set secrets (Replit Secrets panel, never in code)

| Secret | Required? |
|---|---|
| `ANTHROPIC_API_KEY` | Yes — needed for extraction |
| `TABPFN_TOKEN` | No — omit to skip the optional TabPFN signal entirely |

## 3. Smoke-test it works, before wiring anything up

```bash
python3 -c "
from market_lens.schema_loader import load_schema
from market_lens.evidence import build_evidence
import sqlite3

schema = load_schema()
con = sqlite3.connect('market_table_combined/market.sqlite')
con.row_factory = sqlite3.Row
row = dict(con.execute('SELECT * FROM ndas LIMIT 1').fetchone())
target = {fid: row.get(fid) for fid in schema.field_ids}
bundle = build_evidence(schema, target, 'market_table_combined')
print(bundle['off_market_index'], len(bundle['fields']), 'fields scored')
"
```

Should print a number and `18 fields scored`. If it does, the table and
library are wired correctly.

## 4. What you're building on top of this

This package does extraction + statistics. **It does not write the report.**
Your job, roughly:

```python
from market_lens.extract import extract_file
from market_lens.schema_loader import load_schema, coerce_row
from market_lens.evidence import build_evidence

schema = load_schema()
ext = extract_file("uploaded_nda.pdf", schema=schema)      # -> typed row + evidence quotes
target = coerce_row(schema, ext.row)
bundle = build_evidence(schema, target, "market_table_combined")

# YOUR CODE: one LLM call with `ext`, `bundle`, and the doc text ->
# the human-facing Market Position Report. See spec.md §3 and §6 for
# exactly what to put in that prompt and why (the two rarity signals
# often disagree on purpose — that disagreement is itself useful context).
```

## 5. Two things not to get wrong

- **Don't show raw scores to the end client.** They're for the reviewing
  lawyer. See spec.md §8 for the recommended lawyer-view/client-view split.
- **Don't call TabPFN's `fit_all_fields()` per request.** It's expensive
  (18 network calls, re-uploads the whole population). Fit once at startup
  or on a schedule, keep the result in memory, call `score_new_doc()` (cheap)
  per incoming NDA. See spec.md §7.

If anything here is ambiguous, spec.md has the full reasoning — it was
written by the same team that built this library, specifically for this
handoff.
