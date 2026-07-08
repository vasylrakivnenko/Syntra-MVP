"""Whole-document NDA extraction (D3).

One Claude call per NDA, structured output enforced against the schema-derived
JSON Schema. No chunking (fixed schema + known doc type). Emits one typed row
plus evidence spans and a coverage metric.

Determinism note: the modern models reject `temperature` (400). The substitute
for temp-0 extraction is structured outputs + thinking disabled, which is what
this uses.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile). This is the one
leg of the pipeline that needs the API; everything downstream runs offline on
the emitted rows.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from .schema_loader import Schema, build_json_schema, load_schema

# Workhorse extraction model. Swap to "claude-opus-4-8" if the golden-set gate
# shows the long-tail (rare) fields need it — Sonnet 5 is the default because
# structured whole-doc extraction is in its wheelhouse and per-doc cost matters
# at 5k+ scale.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000

SYSTEM_PROMPT = """\
You are a precise contract-analysis engine extracting typed fields from a \
Non-Disclosure Agreement into a fixed schema. For every field:

- Set `value` to your reading of the clause.
- CRITICAL — distinguish two kinds of "no":
    * value = false (for booleans) means the clause is genuinely ABSENT from the document.
    * value = null means you COULD NOT DETERMINE it from the text (ambiguous, unreadable, cut off).
  Never guess. If unsure, use null, not false.
- `evidence_span` must be a short VERBATIM quote from the document that supports \
the value, or null if the value is false/null (nothing to quote).
- numeric_months: normalize all durations to months (2 years -> 24). \
Perpetual/indefinite -> 9999.
- enum fields: choose exactly one of the allowed values (or null).

Return only the structured object."""


@dataclass
class Extraction:
    row: dict[str, Any]  # field_id -> value  (flat, for the market table)
    evidence: dict[str, str | None]  # field_id -> supporting quote
    meta: dict[str, Any] = dc_field(default_factory=dict)


def _doc_block(path: Path) -> dict[str, Any]:
    """Build the user content block for a .txt or .pdf NDA."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        }
    # treat everything else as UTF-8 text
    return {"type": "text", "text": path.read_text(encoding="utf-8", errors="replace")}


def extract_file(
    path: Path | str,
    *,
    schema: Schema | None = None,
    license_class: str = "unknown",
    client: Any | None = None,
) -> Extraction:
    """Extract one NDA file into a typed row. `client` is injectable for tests;
    defaults to a fresh anthropic.Anthropic()."""
    path = Path(path)
    schema = schema or load_schema()
    json_schema = build_json_schema(schema)

    if client is None:
        import anthropic  # imported lazily so the offline pipeline needs no SDK

        client = anthropic.Anthropic()

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": json_schema}},
        messages=[
            {
                "role": "user",
                "content": [
                    _doc_block(path),
                    {"type": "text", "text": "Extract the NDA fields per the schema."},
                ],
            }
        ],
    )

    if resp.stop_reason == "refusal":
        raise RuntimeError(f"extraction refused for {path.name}: {resp.stop_details}")

    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if text is None:
        raise RuntimeError(f"no text block in response for {path.name}")

    parsed = json.loads(text)
    return _to_extraction(parsed, schema, path, license_class, model=MODEL)


def _to_extraction(
    parsed: dict[str, Any],
    schema: Schema,
    path: Path,
    license_class: str,
    model: str,
) -> Extraction:
    row: dict[str, Any] = {}
    evidence: dict[str, str | None] = {}
    determinable = 0
    for f in schema.fields:
        cell = parsed.get(f.id) or {}
        value = cell.get("value")
        row[f.id] = value
        evidence[f.id] = cell.get("evidence_span")
        if value is not None:
            determinable += 1

    coverage = determinable / len(schema.fields) if schema.fields else 0.0
    meta = {
        "source_doc": path.name,
        "schema_version": schema.version,
        "model": model,
        "license_class": license_class,  # gates inclusion in the asset table (D5)
        "coverage": round(coverage, 4),
    }
    return Extraction(row=row, evidence=evidence, meta=meta)


def extraction_to_record(ext: Extraction) -> dict[str, Any]:
    """Flatten to a single JSON-serialisable record for on-disk storage."""
    return {**ext.row, "_meta": ext.meta, "_evidence": ext.evidence}
