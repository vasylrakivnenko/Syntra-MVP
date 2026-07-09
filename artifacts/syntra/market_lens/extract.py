"""Whole-document NDA extraction (D3).

One or more Claude calls per NDA (batched — see FIELD_BATCH_SIZE), via
Anthropic's `messages.parse` structured-output path: a pydantic model built
from the schema (schema_loader.build_extraction_model) is passed as
`output_format` and the SDK returns a validated, typed object directly (no
hand-rolled JSON Schema, no manual json.loads). No chunking within a batch
(fixed schema + known doc type). Emits one typed row plus evidence spans and
a coverage metric.

null-vs-absent: bool/enum fields are extracted as a flat Literal
(true/false/undetermined, or the enum values plus "undetermined") rather than
a nullable type — see schema_loader.UNDETERMINED for why. `_canonicalize_value`
below converts that wire-level tri-state back into the Python True/False/None
(or enum-string/None) shape every downstream consumer expects, so nothing
past this module needs to know the wire encoding changed.

Batching note: Anthropic's structured-outputs validator caps a schema at 16
union/nullable-typed parameters. The flat-Literal encoding above means only
numeric_months fields are nullable now (2 of 18 in the current schema) — well
under the cap — so FIELD_BATCH_SIZE no longer needs to split the schema into
several calls the way the old nullable-everything JSON Schema did. This is
UNVERIFIED against a live call (no API key in the dev sandbox this was written
in); if Anthropic rejects a full-schema call for some other reason, lower
FIELD_BATCH_SIZE back down — the batching machinery is unchanged.

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

from .schema_loader import Field, Schema, UNDETERMINED, build_extraction_model, load_schema

# Workhorse extraction model. Swap to "claude-opus-4-8" if the golden-set gate
# shows the long-tail (rare) fields need it — Sonnet 5 is the default because
# structured whole-doc extraction is in its wheelhouse and per-doc cost matters
# at 5k+ scale.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000
# Fields per call. Set to the full schema width: the flat-Literal encoding
# (see module docstring) leaves only 2 of 18 fields nullable, well under
# Anthropic's 16-union-typed-parameter cap, so one call should now cover the
# whole schema. Lower this if a live run says otherwise.
FIELD_BATCH_SIZE = 18

SYSTEM_PROMPT = """\
You are a precise contract-analysis engine extracting typed fields from a \
Non-Disclosure Agreement into a fixed schema. For every field:

- Boolean-type fields take exactly one of three literal strings — never guess
  between the last two:
    * "true"          — the clause is present in the document.
    * "false"         — the clause is explicitly ABSENT from the document.
    * "undetermined"  — the text does not let you tell either way (ambiguous, unreadable, cut off).
- Enum-type fields: choose exactly one of the allowed values, or "undetermined" \
if the document does not let you determine it.
- numeric_months fields: normalize all durations to months (2 years -> 24); \
perpetual/indefinite -> 9999; null if it cannot be determined from the text.
- `evidence_span` must be a short VERBATIM quote from the document that supports \
the value, or an empty string if there is nothing to quote (false/undetermined/null values).

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


def _batch_fields(fields: list[Field], batch_size: int) -> list[list[Field]]:
    return [fields[i : i + batch_size] for i in range(0, len(fields), batch_size)]


def _canonicalize_value(f: Field, raw: Any) -> Any:
    """Undo the wire-level tri-state Literal encoding (see module docstring)
    back into the canonical shape schema_loader.coerce_row / stats / build_table
    already expect: Python True/False/None for bool, the enum string or None
    for enum. numeric_months is already Optional[float] on the wire — passed
    through unchanged."""
    if f.type == "bool":
        return {"true": True, "false": False, UNDETERMINED: None}[raw]
    if f.type == "enum":
        return None if raw == UNDETERMINED else raw
    return raw


def _extract_batch(
    client: Any, doc_block: dict[str, Any], schema: Schema, fields: list[Field], doc_label: str
) -> dict[str, Any]:
    model_cls = build_extraction_model(schema, fields=fields)
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=SYSTEM_PROMPT,
        output_format=model_cls,
        messages=[
            {
                "role": "user",
                "content": [
                    doc_block,
                    {"type": "text", "text": "Extract the NDA fields per the schema."},
                ],
            }
        ],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"extraction refused for {doc_label}")
    parsed_output = next(
        (b.parsed_output for b in resp.content if getattr(b, "type", None) == "text"), None
    )
    if parsed_output is None:
        raise RuntimeError(f"no parsed output in response for {doc_label}")
    parsed = parsed_output.model_dump()
    for f in fields:
        parsed[f.id]["value"] = _canonicalize_value(f, parsed[f.id]["value"])
    return parsed


def extract_text(
    document_text: str,
    *,
    source_name: str = "<text>",
    schema: Schema | None = None,
    license_class: str = "unknown",
    client: Any | None = None,
) -> Extraction:
    """Extract one NDA from raw text into a typed row (mirrors the
    providers.azure_openai / providers.hyperbolic API). `client` is injectable
    for tests; defaults to a fresh anthropic.Anthropic(). Issues one call per
    field batch (see FIELD_BATCH_SIZE) and merges the results."""
    schema = schema or load_schema()
    doc_block = {"type": "text", "text": document_text}

    if client is None:
        import anthropic  # imported lazily so the offline pipeline needs no SDK

        client = anthropic.Anthropic()

    parsed: dict[str, Any] = {}
    for batch in _batch_fields(schema.fields, FIELD_BATCH_SIZE):
        parsed.update(_extract_batch(client, doc_block, schema, batch, source_name))

    return _to_extraction(parsed, schema, Path(source_name), license_class, model=MODEL)


def extract_file(
    path: Path | str,
    *,
    schema: Schema | None = None,
    license_class: str = "unknown",
    client: Any | None = None,
) -> Extraction:
    """Extract one NDA file (.txt or .pdf) into a typed row."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        schema = schema or load_schema()
        doc_block = _doc_block(path)
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        parsed: dict[str, Any] = {}
        for batch in _batch_fields(schema.fields, FIELD_BATCH_SIZE):
            parsed.update(_extract_batch(client, doc_block, schema, batch, path.name))
        return _to_extraction(parsed, schema, path, license_class, model=MODEL)
    return extract_text(
        path.read_text(encoding="utf-8", errors="replace"),
        source_name=path.name, schema=schema, license_class=license_class, client=client,
    )


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
        evidence[f.id] = cell.get("evidence_span") or None  # "" sentinel -> None
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


def save_extraction(
    ext: Extraction,
    record_path: Path,
    *,
    document_text: str,
    text_dir: Path | None = None,
) -> None:
    """Persist one extraction: the JSON record at record_path, AND the full
    source document text as a sibling file (same stem, .txt) -- so a result
    can be re-verified or re-extracted later without re-fetching the URL.

    Requires ext.meta['source_url'] to already be set: every stored record
    should carry both the source URL and the full text it came from, not one
    or the other. Provider-agnostic -- works with records from extract.py,
    providers/azure_openai.py, or providers/hyperbolic.py alike.

    text_dir defaults to a `source_texts/` directory as a SIBLING of
    record_path's parent (e.g. records_foo/bar.json -> source_texts/bar.txt),
    i.e. one shared cache at the project root, not nested per-corpus.
    """
    if not ext.meta.get("source_url"):
        raise ValueError(f"refusing to save {record_path.name}: ext.meta['source_url'] is not set")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(extraction_to_record(ext), indent=2))
    text_dir = text_dir or (record_path.parent.parent / "source_texts")
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / f"{record_path.stem}.txt").write_text(document_text)
