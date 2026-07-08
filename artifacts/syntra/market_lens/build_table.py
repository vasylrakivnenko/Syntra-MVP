"""Build the market table (D5) from extracted records.

Input: a directory of *.json extraction records (one per NDA), as written by
`extract_dir`. Output: a normalized parquet + SQLite table.

License enforcement (the moat guard): rows tagged license_class == "eval_only"
(e.g. ContractNLI-derived) are EXCLUDED from the asset-eligible table. They stay
available for golden-set / eval use but never feed the commercial statistics.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .schema_loader import Schema, load_schema

EVAL_ONLY = "eval_only"


def extract_dir(
    src_dir: Path | str,
    out_dir: Path | str,
    *,
    license_class: str = "unknown",
    schema: Schema | None = None,
    glob: str = "*",
) -> list[Path]:
    """Extract every NDA file in src_dir -> one JSON record per doc in out_dir.
    Returns the written record paths. (This is the leg that calls the API.)"""
    from .extract import extract_file, extraction_to_record  # lazy: needs SDK

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = schema or load_schema()
    written: list[Path] = []
    for doc in sorted(src_dir.glob(glob)):
        if doc.is_dir() or doc.suffix.lower() not in {".txt", ".pdf", ".md"}:
            continue
        ext = extract_file(doc, schema=schema, license_class=license_class)
        rec_path = out_dir / f"{doc.stem}.json"
        rec_path.write_text(json.dumps(extraction_to_record(ext), indent=2))
        written.append(rec_path)
        print(f"  extracted {doc.name}  (coverage {ext.meta['coverage']:.0%})")
    return written


def load_records(records_dir: Path | str) -> list[dict[str, Any]]:
    recs = []
    for p in sorted(Path(records_dir).glob("*.json")):
        recs.append(json.loads(p.read_text()))
    return recs


def build_table(
    records_dir: Path | str,
    out_dir: Path | str,
    *,
    schema: Schema | None = None,
) -> dict[str, Any]:
    """Records -> asset-eligible market table (parquet + sqlite). Returns a
    summary dict. eval_only rows are dropped with a logged count (never silent)."""
    schema = schema or load_schema()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(records_dir)
    rows: list[dict[str, Any]] = []
    dropped_eval_only = 0
    for rec in records:
        meta = rec.get("_meta", {})
        if meta.get("license_class") == EVAL_ONLY:
            dropped_eval_only += 1
            continue
        row = {fid: rec.get(fid) for fid in schema.field_ids}
        row["_source_doc"] = meta.get("source_doc")
        row["_schema_version"] = meta.get("schema_version")
        row["_license_class"] = meta.get("license_class")
        rows.append(row)

    _write_sqlite(rows, schema, out_dir / "market.sqlite")
    parquet_ok = _try_write_parquet(rows, out_dir / "market.parquet")

    summary = {
        "records_seen": len(records),
        "rows_in_table": len(rows),
        "dropped_eval_only": dropped_eval_only,
        "schema_version": schema.version,
        "parquet_written": parquet_ok,
        "sqlite_path": str(out_dir / "market.sqlite"),
    }
    if dropped_eval_only:
        print(f"  license guard: dropped {dropped_eval_only} eval_only row(s) from the asset table")
    return summary


def _write_sqlite(rows: list[dict[str, Any]], schema: Schema, path: Path) -> None:
    cols = schema.field_ids + ["_source_doc", "_schema_version", "_license_class"]
    con = sqlite3.connect(path)
    try:
        con.execute("DROP TABLE IF EXISTS ndas")
        col_defs = ", ".join(f'"{c}"' for c in cols)
        con.execute(f"CREATE TABLE ndas ({col_defs})")
        placeholders = ", ".join("?" for _ in cols)
        con.executemany(
            f"INSERT INTO ndas VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        con.commit()
    finally:
        con.close()


def _try_write_parquet(rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    pd.DataFrame(rows).to_parquet(path, index=False)
    return True


def _main() -> None:
    ap = argparse.ArgumentParser(description="Build the Market Lens NDA table.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="extract a directory of NDAs -> records")
    e.add_argument("src_dir")
    e.add_argument("out_dir")
    e.add_argument("--license-class", default="unknown",
                   help="asset_eligible | eval_only | unknown (default)")

    b = sub.add_parser("build", help="records dir -> market table")
    b.add_argument("records_dir")
    b.add_argument("out_dir")

    args = ap.parse_args()
    if args.cmd == "extract":
        paths = extract_dir(args.src_dir, args.out_dir, license_class=args.license_class)
        print(f"wrote {len(paths)} record(s) to {args.out_dir}")
    elif args.cmd == "build":
        summary = build_table(args.records_dir, args.out_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
