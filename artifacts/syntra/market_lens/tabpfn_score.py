"""Live TabPFN per-field scoring for a NEW NDA (one not already in the corpus).

Two operations, with very different costs -- see spec.md for the full
architecture rationale:

  fit_all_fields()  EXPENSIVE. Uploads the whole population (~900+ rows per
                     field) to TabPFN's cloud, once per field (18 network
                     calls). Call this ONCE per process lifetime, or on a
                     schedule whenever the market table is rebuilt -- NEVER
                     per scoring request. Returns a dict of fitted
                     classifiers; keep it in memory (module-level singleton /
                     long-lived process) and reuse it.

  score_new_doc()    CHEAP. Sends just the one new document to each field's
                     ALREADY-FITTED classifier and gets back the probability
                     TabPFN assigns to that field's actual value, given the
                     other 17 fields. This is the one that runs per request.

This mirrors exactly the fit-once/predict-many pattern already proven in
scripts/tabpfn_pilot.py's run_probe() stage (fit on the whole population,
predict for docs that were never part of training) -- just pointed at a real
new document instead of a synthetic probe.

Requires TABPFN_TOKEN in the environment and the `tabpfn_client` package.
Never touches thinking mode (that's a separate, quota-capped feature this
module doesn't use) -- plain (non-thinking) fit/predict has no known hard
monthly cap, only per-call cost (see TabPFN's pricing page).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .schema_loader import Schema, bucket_numeric, load_schema


def _row_to_frame_dict(schema: Schema, row: dict[str, Any]) -> dict[str, Any]:
    """Same categorical-string encoding scripts/tabpfn_pilot.py's to_frame
    uses: one string column per field, None for missing -- bool/enum pass
    through as strings, numeric_months gets bucketed (never raw percentile;
    see market_lens/evidence.py's docstring for why that matters for the
    perpetual-duration sentinel)."""
    out: dict[str, Any] = {}
    for f in schema.fields:
        v = row.get(f.id)
        if v is None:
            out[f.id] = None
        elif f.type == "bool":
            out[f.id] = "true" if v else "false"
        elif f.type == "enum":
            out[f.id] = str(v)
        else:
            out[f.id] = bucket_numeric(f, float(v))
    return out


def load_population_frame(table_dir: Path | str, schema: Schema | None = None) -> pd.DataFrame:
    """The same deduped, license-clean population market_lens.score and
    scripts/tabpfn_pilot.py score against -- read directly from the built
    market table, not re-derived here."""
    schema = schema or load_schema()
    con = sqlite3.connect(Path(table_dir) / "market.sqlite")
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute("SELECT * FROM ndas")]
    finally:
        con.close()
    frame_rows = {
        (r.get("_source_doc") or f"row{i}"): _row_to_frame_dict(schema, r)
        for i, r in enumerate(rows)
    }
    return pd.DataFrame.from_dict(frame_rows, orient="index")


def fit_all_fields(table_dir: Path | str, schema: Schema | None = None) -> dict[str, Any]:
    """Fit one TabPFNClassifier per field against the WHOLE population.
    EXPENSIVE (see module docstring) -- call once, keep the result, reuse it.
    Fields with fewer than 2 observed classes are skipped (nothing to
    predict) and simply absent from the returned dict."""
    from tabpfn_client import TabPFNClassifier, set_access_token

    set_access_token(os.environ["TABPFN_TOKEN"])
    schema = schema or load_schema()
    df = load_population_frame(table_dir, schema)

    fitted: dict[str, Any] = {}
    for f in schema.fields:
        mask = df[f.id].notna()
        y = df.loc[mask, f.id]
        if y.nunique() < 2:
            continue
        X = df.loc[mask, [c for c in df.columns if c != f.id]]
        clf = TabPFNClassifier()
        clf.fit(X, y)
        fitted[f.id] = clf
    return fitted


def score_new_doc(
    schema: Schema, fitted: dict[str, Any], target_row: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """CHEAP: for every field with a fitted classifier and a value on this
    doc, the probability that classifier assigns to the doc's actual value
    -- i.e. "given the other 17 fields, how likely was this one". Returns
    {field_id: {"value": <original value>, "p_obs": float | None}};
    p_obs is None where there's no fitted classifier for that field or the
    doc's own value is missing (nothing to condition on)."""
    row = _row_to_frame_dict(schema, target_row)
    out: dict[str, dict[str, Any]] = {}
    for f in schema.fields:
        v = row.get(f.id)
        clf = fitted.get(f.id)
        if v is None or clf is None:
            out[f.id] = {"value": target_row.get(f.id), "p_obs": None}
            continue
        x_row = pd.DataFrame([{k: vv for k, vv in row.items() if k != f.id}])
        classes = list(clf.classes_)
        proba = clf.predict_proba(x_row)[0]
        p = float(proba[classes.index(v)]) if v in classes else 0.0
        out[f.id] = {"value": target_row.get(f.id), "p_obs": p}
    return out
