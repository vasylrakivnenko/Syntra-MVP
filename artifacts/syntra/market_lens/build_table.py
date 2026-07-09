"""Build the market table (D5) from extracted records.

Input: one or more directories of *.json extraction records (one per NDA), as
written by `extract_dir`. Output: a normalized parquet + SQLite table.

License enforcement (the moat guard) is an ALLOWLIST, not a blocklist: only
rows tagged with a verified-clean license_class (see ASSET_ELIGIBLE_LICENSE_
CLASSES) make it into the asset table. Anything else -- "eval_only"
(ContractNLI-derived under the old, since-corrected belief), "unknown"
(pre-license-discipline test extractions), "crosscheck_only" (dual-model
comparison re-extractions of docs already counted elsewhere), or any future
class nobody's vetted yet -- is dropped. An allowlist is the safe default
here: a NEW license_class showing up in a record (typo, new corpus, forgotten
tag) is excluded until someone deliberately adds it, rather than silently
included until someone notices. See MEMORY market-lens-data-licensing.

Cross-directory dedup: independent harvests can find the same public filing
(e.g. EDGAR full-text search and the Material Contracts Corpus both indexing
the same SEC exhibit) -- see build_table's docstring below.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .schema_loader import Schema, load_schema

ASSET_ELIGIBLE_LICENSE_CLASSES = {"asset_eligible", "cc_by_4.0"}


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
    records_dir: Path | str | list[Path | str],
    out_dir: Path | str,
    *,
    schema: Schema | None = None,
) -> dict[str, Any]:
    """One or more records dirs -> asset-eligible market table (parquet +
    sqlite). Returns a summary dict. Rows are dropped (never silently), for
    two independent reasons, each counted separately in the summary:

      - license_class not in ASSET_ELIGIBLE_LICENSE_CLASSES (see module
        docstring -- this is an allowlist, not a blocklist)
      - a duplicate source_url already claimed by an earlier row. Directories
        are processed in the order given; the FIRST occurrence of a
        source_url wins and later ones are dropped. Independent harvests
        (e.g. an EDGAR full-text search and the Material Contracts Corpus)
        can and do find the same public filing -- without this, the same
        document would be double-counted in every statistic downstream.
    """
    schema = schema or load_schema()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = [records_dir] if isinstance(records_dir, (str, Path)) else list(records_dir)

    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    dropped_by_license: dict[str, int] = {}
    dropped_duplicate = 0
    records_seen = 0

    for d in dirs:
        for rec in load_records(d):
            records_seen += 1
            meta = rec.get("_meta", {})
            lic = meta.get("license_class")
            if lic not in ASSET_ELIGIBLE_LICENSE_CLASSES:
                dropped_by_license[lic] = dropped_by_license.get(lic, 0) + 1
                continue

            url = meta.get("source_url")
            if url:
                if url in seen_urls:
                    dropped_duplicate += 1
                    continue
                seen_urls.add(url)

            row = {fid: rec.get(fid) for fid in schema.field_ids}
            row["_source_doc"] = meta.get("source_doc")
            row["_schema_version"] = meta.get("schema_version")
            row["_license_class"] = lic
            rows.append(row)

    _write_sqlite(rows, schema, out_dir / "market.sqlite")
    parquet_ok = _try_write_parquet(rows, out_dir / "market.parquet")

    # Persist the Off-Market reference so NEW docs can be scored with v1.1
    # (calibrated pvalue statistic) without re-scoring the corpus.
    from .schema_loader import coerce_row
    from .stats import REFERENCE_FILENAME, build_reference

    ref = build_reference(schema, [coerce_row(schema, r) for r in rows])
    (out_dir / REFERENCE_FILENAME).write_text(json.dumps(ref))

    summary = {
        "source_dirs": [str(d) for d in dirs],
        "records_seen": records_seen,
        "rows_in_table": len(rows),
        "dropped_by_license_class": dropped_by_license,
        "dropped_duplicate_source_url": dropped_duplicate,
        "schema_version": schema.version,
        "parquet_written": parquet_ok,
        "sqlite_path": str(out_dir / "market.sqlite"),
        "reference_segments": {k: v["n"] for k, v in ref["segments"].items()},
    }
    if dropped_by_license:
        print(f"  license guard: dropped {sum(dropped_by_license.values())} row(s) "
              f"by class: {dropped_by_license}")
    if dropped_duplicate:
        print(f"  dedup: dropped {dropped_duplicate} row(s) sharing a source_url "
              f"with an earlier-processed row")
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

    b = sub.add_parser("build", help="one or more records dirs -> market table")
    b.add_argument("records_dirs", nargs="+", help="one or more records directories")
    b.add_argument("--out-dir", required=True)

    args = ap.parse_args()
    if args.cmd == "extract":
        paths = extract_dir(args.src_dir, args.out_dir, license_class=args.license_class)
        print(f"wrote {len(paths)} record(s) to {args.out_dir}")
    elif args.cmd == "build":
        summary = build_table(args.records_dirs, args.out_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
