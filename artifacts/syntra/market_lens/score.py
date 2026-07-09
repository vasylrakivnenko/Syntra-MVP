"""Score one NDA against the market table — the whole no-UI "product".

    python -m market_lens.score path/to/nda.pdf --table ./market_table
    python -m market_lens.score --record ./records/foo.json --table ./market_table

Prints:
  (a) per-field position vs. the segment (share of segment sharing this exact
      value for bool/enum, or this same bucket_numeric bucket for numeric --
      never raw percentile; see stats.univariate's docstring for why), and
  (b) the Off-Market Index + the rarest clause COMBINATIONS driving it, each as
      an explainable "X of N docs in segment" string.

Segment = same mutuality as the target (mutual vs. unilateral).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema_loader import PERPETUAL_SENTINEL, Schema, bucket_numeric, coerce_row, load_schema
from .stats import (
    OFF_MARKET_THRESHOLD,
    off_market_index,
    score_against_reference,
    segment,
    univariate,
)


def _load_target(args: argparse.Namespace, schema: Schema) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (row, meta) for the target, from a pre-extracted --record or a live
    extraction of --file."""
    if args.record:
        rec = json.loads(Path(args.record).read_text())
        row = {fid: rec.get(fid) for fid in schema.field_ids}
        return coerce_row(schema, row), rec.get("_meta", {})
    from .extract import extract_file  # lazy: needs SDK + key

    ext = extract_file(args.file, schema=schema, license_class=args.license_class)
    return coerce_row(schema, ext.row), ext.meta


def _fmt_value(v: Any) -> str:
    if v is None:
        return "null (undetermined)"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        if v >= PERPETUAL_SENTINEL:
            return "perpetual"
        return f"{v:g}"  # drop trailing .0 on integral months
    return str(v)


def render(target_row: dict[str, Any], meta: dict[str, Any],
           table_dir: Path | None, schema: Schema) -> str:
    lines: list[str] = []
    doc = meta.get("source_doc", "<target>")
    lines.append(f"NDA: {doc}")
    lines.append(f"schema {meta.get('schema_version', schema.version)} | "
                 f"model {meta.get('model', '-')} | coverage {meta.get('coverage', '-')}")

    if table_dir is None:
        lines.append("\n(no market table given — extracted row only)\n")
        for f in schema.fields:
            lines.append(f"  {f.id:<32} {_fmt_value(target_row.get(f.id))}")
        return "\n".join(lines)

    from .build_table import load_records  # local import to avoid cycle at import time

    rows = _load_table_rows(table_dir, schema)
    mutual = target_row.get("mutual")
    seg = segment(rows, mutual)
    seg_label = ("mutual" if mutual is True else "unilateral" if mutual is False
                 else "whole population (mutuality undetermined)")

    lines.append(f"\nSegment: {seg_label}  (N = {len(seg)} of {len(rows)} in table)")

    # --- (a) univariate parity ---
    uni = univariate(schema, seg)
    lines.append("\nPer-field vs. segment")
    lines.append("  " + "-" * 68)
    for f in schema.fields:
        tv = target_row.get(f.id)
        stat = uni[f.id]
        if tv is None:
            pos = "—"
        elif f.type in ("bool", "enum"):
            key = "true" if tv is True else "false" if tv is False else str(tv)
            share = stat["freq"].get(key, 0.0)
            pos = f"{share:5.0%} of segment share this value"
        else:  # numeric -- bucket share, NOT raw percentile (see stats.univariate docstring:
                # percentile would force every perpetual-duration doc to the 100th percentile
                # regardless of how common perpetual actually is in the segment)
            b = bucket_numeric(f, float(tv))
            share = stat.get("bucket_freq", {}).get(b, 0.0)
            pos = f"{share:5.0%} of segment share bucket [{b}] ({tv:g} mo)"
        lines.append(f"  {f.id:<32} {_fmt_value(tv):<22} {pos}")

    # --- (b) Off-Market Index ---
    ref = _load_reference(table_dir)
    if ref is not None:
        om = score_against_reference(schema, target_row, ref)
        lines.append("\nOff-Market Index (v1.1, vs persisted reference population)")
        lines.append("  " + "-" * 68)
        if om is None:
            lines.append("  INSUFFICIENT DATA — target's segment below the N floor in the reference.")
        else:
            lines.append(f"  Index: {om.index}/100   (rank vs reference; raw surprise {om.raw})")
            lines.append("  Most surprising clause combinations in this NDA:")
            for contrib in om.contributions:
                if len(contrib) == 4:  # pvalue statistic
                    combo, obs, exp, pval = contrib
                    lines.append(f"    expected ~{exp:.1f}, observed {obs} (p={pval:.2g})   "
                                 + "  +  ".join(combo))
                else:
                    combo, obs, exp = contrib
                    lines.append(f"    expected ~{exp:.1f}, observed {obs}   " + "  +  ".join(combo))
            lines.append("\n  (statistical context, not legal advice)")
        return "\n".join(lines)

    # Legacy fallback: table has no persisted reference (v1 raw-rarity index —
    # known to saturate; rebuild the table to get the v1.1 reference).
    res = off_market_index(schema, target_row, seg)
    lines.append("\nOff-Market Index (LEGACY v1 — no reference found; rebuild table for v1.1)")
    lines.append("  " + "-" * 68)
    if res.status == "insufficient_data":
        lines.append(f"  INSUFFICIENT DATA — segment N={res.segment_n} below floor "
                     f"or no anchorable clause combinations. Not scored.")
    else:
        lines.append(f"  Index: {res.index}/100   (higher = rarer clause combination)")
        lines.append(f"  Rarest clause combinations present in this NDA:")
        for c in res.contributions:
            flag = "  << OFF-MARKET" if c.off_market else ""
            combo = "  +  ".join(c.combo)
            lines.append(f"    {c.support:5.1%} of segment ({c.count}/{res.segment_n})"
                         f"   {combo}{flag}")
        lines.append(f"\n  (off-market flag = combination in <= {OFF_MARKET_THRESHOLD:.0%} of segment; "
                     f"statistical context, not legal advice)")
    return "\n".join(lines)


def _load_reference(table_dir: Path | None) -> dict | None:
    if table_dir is None:
        return None
    ref_path = table_dir / "omx_reference.json"
    if not ref_path.exists():
        return None
    return json.loads(ref_path.read_text())


def _load_table_rows(table_dir: Path, schema: Schema) -> list[dict[str, Any]]:
    """Prefer the built parquet/sqlite; fall back to raw records dir."""
    import sqlite3

    sqlite_path = table_dir / "market.sqlite"
    if sqlite_path.exists():
        con = sqlite3.connect(sqlite_path)
        try:
            con.row_factory = sqlite3.Row
            cur = con.execute("SELECT * FROM ndas")
            return [coerce_row(schema, dict(r)) for r in cur.fetchall()]
        finally:
            con.close()
    # fall back to a directory of extraction records
    from .build_table import load_records

    recs = load_records(table_dir)
    return [coerce_row(schema, {fid: r.get(fid) for fid in schema.field_ids}) for r in recs]


def _main() -> None:
    ap = argparse.ArgumentParser(description="Score an NDA against the market.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="NDA file (.pdf/.txt/.md) to extract and score")
    src.add_argument("--record", help="score a pre-extracted JSON record (no API call)")
    ap.add_argument("--table", help="market table dir (parquet/sqlite or records)")
    ap.add_argument("--license-class", default="unknown")
    args = ap.parse_args()

    schema = load_schema()
    target_row, meta = _load_target(args, schema)
    table_dir = Path(args.table) if args.table else None
    print(render(target_row, meta, table_dir, schema))


if __name__ == "__main__":
    _main()
