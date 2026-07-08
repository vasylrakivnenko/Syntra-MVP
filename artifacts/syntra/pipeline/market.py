"""Market Lens adapter — the one bridge between Syntra and the vendored
market_lens library (see market_lens/ and market_data/).

Two responsibilities, nothing else:
  1. extract_market_row  — whole-doc NDA extraction via Syntra's existing
     OpenAI-compatible client (llm.py), reusing market_lens's own prompt,
     schema and Extraction contract. No new API keys, no anthropic SDK.
  2. score_market_row    — score the extracted row against the shipped
     200-NDA market table (SQLite, offline, no API calls).

run_market_lens() composes both into one JSON-serialisable report that the
contract page renders. Everything here is ADVISORY statistical context —
it never feeds routing or triage (the Off-Market Index is unvalidated;
see market_lens's own caveats).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from llm import MODEL, get_client
from market_lens.extract import Extraction, _to_extraction
from market_lens.providers.hyperbolic import (
    SYSTEM_PROMPT,
    _extract_json_object,
    _field_guide,
)
from market_lens.schema_loader import (
    PERPETUAL_SENTINEL,
    Schema,
    coerce_row,
    load_schema,
)
from market_lens.stats import off_market_index, segment as market_segment, univariate

MARKET_TABLE = Path(__file__).resolve().parent.parent / "market_data" / "market.sqlite"

# Extraction can time out on very long documents; cap the text we send.
_MAX_DOC_CHARS = 60_000


# ── extraction (via Syntra's LLM client) ─────────────────────────────────────

def extract_market_row(text: str, source_name: str = "<doc>",
                       schema: Schema | None = None) -> Extraction:
    """One whole-doc extraction call returning market_lens's Extraction
    (row + evidence + meta), transported over Syntra's OpenAI client."""
    schema = schema or load_schema()
    user = (
        "Schema fields:\n" + _field_guide(schema)
        + "\n\nReturn a JSON object whose keys are EXACTLY these field ids, each "
          'mapping to {"value": ..., "evidence_span": ...}.\n\n'
        + "=== NDA DOCUMENT ===\n" + text[:_MAX_DOC_CHARS]
    )
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    parsed = _extract_json_object(resp.choices[0].message.content or "")
    return _to_extraction(parsed, schema, Path(source_name),
                          license_class="internal", model=MODEL)


# ── scoring (offline, against the shipped table) ─────────────────────────────

def _load_market_rows(schema: Schema) -> list[dict[str, Any]]:
    con = sqlite3.connect(MARKET_TABLE)
    con.row_factory = sqlite3.Row
    try:
        return [coerce_row(schema, dict(r)) for r in con.execute("SELECT * FROM ndas")]
    finally:
        con.close()


def _fmt_value(field, value: Any) -> str:
    if value is None:
        return "undetermined"
    if value is True:
        return "yes"
    if value is False:
        return "absent"
    if field.type == "numeric_months":
        v = float(value)
        return "perpetual" if v >= PERPETUAL_SENTINEL else f"{v:g} months"
    return str(value).replace("_", " ")


def _field_note(field, value: Any, uni_entry: dict[str, Any]) -> str | None:
    """One friendly sentence placing the doc's value in the segment."""
    if value is None:
        return None
    if field.type in ("bool", "enum"):
        key = "true" if value is True else "false" if value is False else str(value)
        share = uni_entry.get("freq", {}).get(key)
        if share is None:
            return "not seen in this segment"
        return f"matches {round(share * 100)}% of comparable NDAs"
    vals = uni_entry.get("values") or []
    if not vals:
        return None
    below = sum(1 for v in vals if v <= float(value))
    pct = round(100 * below / len(vals))
    return f"{pct}th percentile (longer than {pct}% of comparable NDAs)"


def _combo_label(combo: tuple[str, ...]) -> str:
    parts = []
    for tok in combo:
        fid, _, bucket = tok.partition("=")
        parts.append(f"{fid.replace('_', ' ')} = {bucket.replace('_', ' ')}")
    return " + ".join(parts)


def score_market_row(row: dict[str, Any], meta: dict[str, Any] | None = None,
                     schema: Schema | None = None) -> dict[str, Any]:
    """Compare one typed row against the shipped market table. Pure/offline."""
    schema = schema or load_schema()
    all_rows = _load_market_rows(schema)
    row = coerce_row(schema, row)
    seg_rows = market_segment(all_rows, row.get("mutual"))
    uni = univariate(schema, seg_rows)

    fields = []
    for f in schema.fields:
        value = row.get(f.id)
        fields.append({
            "id": f.id,
            "label": f.id.replace("_", " "),
            "region": f.region,
            "value": _fmt_value(f, value),
            "determined": value is not None,
            "note": _field_note(f, value, uni.get(f.id, {})),
        })

    off = off_market_index(schema, row, seg_rows)
    contributions = [
        {
            "label": _combo_label(c.combo),
            "count": c.count,
            "n": off.segment_n,
            "share": round(c.support * 100, 1),
            "off_market": c.off_market,
        }
        for c in off.contributions
    ]

    return {
        "schema_version": schema.version,
        "coverage": (meta or {}).get("coverage"),
        "model": (meta or {}).get("model"),
        "segment": {
            "mutual": row.get("mutual"),
            "n": len(seg_rows),
            "table_n": len(all_rows),
        },
        "fields": fields,
        "off_market": {
            "index": off.index,
            "status": off.status,
            "contributions": contributions,
            "flagged": [c for c in contributions if c["off_market"]],
        },
    }


# ── composition ───────────────────────────────────────────────────────────────

def run_market_lens(text: str, source_name: str = "<doc>") -> dict[str, Any]:
    """Extract + score one NDA. Raises on failure — callers decide whether
    market context is allowed to fail softly (in the pipeline it is)."""
    schema = load_schema()
    ext = extract_market_row(text, source_name, schema)
    report = score_market_row(ext.row, ext.meta, schema)
    report["evidence"] = {k: v for k, v in ext.evidence.items() if v}
    return report
