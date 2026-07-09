"""Builds the per-field "evidence bundle" for one target NDA -- structured
context for a downstream LLM synthesis step to reason about which clauses
are unusual and why, WITHOUT this module deciding a verdict itself. See
spec.md for the full architecture and where this fits (this is "step 2" --
step 3, the LLM synthesis, is deliberately NOT part of this package).

Two independent signals per field, always both computed, never collapsed
into one number:

  - rule_univariate: what share of comparable NDAs (same mutual/unilateral
    segment) share this exact value. Instant, zero external dependency,
    always available. For numeric_months this is a BUCKET share (see
    schema_loader.bucket_numeric / stats.univariate), never raw percentile.
  - rule_combo: whether this field participates in one of the document's
    rarest 2-/3-way clause COMBINATIONS (market_lens.stats' Off-Market
    Index, v1.1 if the table has a persisted reference, else legacy v1) --
    the existing rules-based product logic, attributed back to the
    individual fields that make it up.
  - tabpfn_p_obs: OPTIONAL. Given the document's other 17 fields, TabPFN's
    estimated probability of this field's actual value. Only present if the
    caller supplies pre-computed scores (see market_lens.tabpfn_score) --
    this module never calls TabPFN itself, so it never fails or slows down
    if TabPFN/TABPFN_TOKEN isn't available.

Every field is included whether or not anything about it looks unusual --
filtering and prioritizing is the synthesis step's job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema_loader import Schema, bucket_numeric
from .score import _load_table_rows
from .stats import off_market_index, score_against_reference, segment, univariate


def _load_reference(table_dir: Path) -> dict | None:
    ref_path = table_dir / "omx_reference.json"
    return json.loads(ref_path.read_text()) if ref_path.exists() else None


def build_evidence(
    schema: Schema,
    target_row: dict[str, Any],
    table_dir: Path | str,
    *,
    tabpfn_p_obs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One evidence bundle for the whole document.

    `tabpfn_p_obs`: the output of market_lens.tabpfn_score.score_new_doc, if
    the caller has it -- pass None (default) to skip TabPFN entirely and get
    a pure rule-based bundle (no external dependency, no possible failure
    from that side).
    """
    table_dir = Path(table_dir)
    rows = _load_table_rows(table_dir, schema)
    mval = target_row.get("mutual")
    seg_rows = segment(rows, mval)
    seg_label = "mutual" if mval is True else "unilateral" if mval is False else "all"
    uni = univariate(schema, seg_rows)

    # Mirrors market_lens.score.render's exact branching: v1.1 (calibrated,
    # persisted reference) if the table has one, else legacy v1 raw-rarity --
    # NOT "try v1.1, silently fall back to legacy on a per-doc miss", since a
    # None from score_against_reference means this doc's segment specifically
    # isn't in the reference (below the N floor), which legacy v1 can't fix.
    ref = _load_reference(table_dir)
    off_market_idx: float | None
    if ref is not None:
        om_ref = score_against_reference(schema, target_row, ref)
        if om_ref is not None:
            off_market_idx = om_ref.index
            off_market_status = "scored"
            combo_contributions = []
            for c in om_ref.contributions:
                if len(c) == 4:  # pvalue statistic
                    combo, obs, exp, pval = c
                    combo_contributions.append(
                        {"combo": list(combo), "observed": obs, "expected": exp, "pvalue": pval})
                else:  # deficit statistic
                    combo, obs, exp = c
                    combo_contributions.append({"combo": list(combo), "observed": obs, "expected": exp})
        else:
            off_market_idx, off_market_status, combo_contributions = None, "insufficient_data", []
    else:
        om_legacy = off_market_index(schema, target_row, seg_rows)
        if om_legacy.status == "scored":
            off_market_idx = om_legacy.index
            off_market_status = "scored"
            combo_contributions = [
                {"combo": list(c.combo), "observed": c.count, "segment_support": c.support}
                for c in om_legacy.contributions
            ]
        else:
            off_market_idx, off_market_status, combo_contributions = None, "insufficient_data", []

    combo_by_field: dict[str, dict[str, Any]] = {}
    for c in combo_contributions:
        for tok in c["combo"]:
            fid = tok.split("=", 1)[0]
            combo_by_field.setdefault(fid, c)  # first = rarest (pre-sorted upstream)

    fields_out: dict[str, Any] = {}
    for f in schema.fields:
        v = target_row.get(f.id)
        entry: dict[str, Any] = {"value": v}
        if v is None:
            entry["rule_univariate"] = None
        elif f.type in ("bool", "enum"):
            key = "true" if v is True else "false" if v is False else str(v)
            entry["rule_univariate"] = {"share": uni[f.id]["freq"].get(key, 0.0)}
        else:
            b = bucket_numeric(f, float(v))
            entry["rule_univariate"] = {"share": uni[f.id].get("bucket_freq", {}).get(b, 0.0), "bucket": b}
        entry["rule_combo"] = combo_by_field.get(f.id)
        entry["tabpfn_p_obs"] = (tabpfn_p_obs or {}).get(f.id, {}).get("p_obs")
        fields_out[f.id] = entry

    return {
        "segment": seg_label,
        "segment_n": len(seg_rows),
        "off_market_index": off_market_idx,
        "off_market_status": off_market_status,
        "fields": fields_out,
    }


def _main() -> None:
    import argparse

    from .schema_loader import coerce_row, load_schema

    ap = argparse.ArgumentParser(description="Build the per-field evidence bundle for one NDA record.")
    ap.add_argument("--record", required=True, help="pre-extracted JSON record (see market_lens.extract)")
    ap.add_argument("--table", required=True, help="market table dir (must contain market.sqlite)")
    args = ap.parse_args()

    schema = load_schema()
    rec = json.loads(Path(args.record).read_text())
    target_row = coerce_row(schema, {fid: rec.get(fid) for fid in schema.field_ids})
    bundle = build_evidence(schema, target_row, args.table)
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    _main()
