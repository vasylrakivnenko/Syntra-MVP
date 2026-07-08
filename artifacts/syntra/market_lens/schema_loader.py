"""Load nda-schema.yaml and derive artifacts the pipeline needs:

  - the JSON Schema for structured-output extraction (enforces null != absent)
  - discrete-token helpers for co-occurrence stats (bool/enum/numeric bucketing)
  - field metadata lookups (type, region, favorability, source)

One schema, three consumers (extract / build_table / stats), so the schema is
loaded here once and nowhere else guesses field types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_PATH = Path(__file__).parent / "schema" / "nda-schema.yaml"
PERPETUAL_SENTINEL = 9999  # numeric_months: perpetual / indefinite


@dataclass(frozen=True)
class Field:
    id: str
    type: str  # bool | enum | numeric_months
    region: str
    favorability: str
    source: str
    enum: list[str] | None
    bins: list[float] | None
    description: str


@dataclass(frozen=True)
class Schema:
    version: str
    fields: list[Field]

    @property
    def field_ids(self) -> list[str]:
        return [f.id for f in self.fields]

    def by_id(self, fid: str) -> Field:
        for f in self.fields:
            if f.id == fid:
                return f
        raise KeyError(fid)


def load_schema(path: Path | str = SCHEMA_PATH) -> Schema:
    raw = yaml.safe_load(Path(path).read_text())
    fields = [
        Field(
            id=f["id"],
            type=f["type"],
            region=f.get("region", "uncategorized"),
            favorability=f.get("favorability", "neutral"),
            source=f.get("source", "unknown"),
            enum=f.get("enum"),
            bins=f.get("bins"),
            description=f.get("description", "").strip(),
        )
        for f in raw["fields"]
    ]
    return Schema(version=str(raw["schema_version"]), fields=fields)


def _value_json_type(field: Field) -> dict[str, Any]:
    """JSON Schema fragment for a single field's `value`. `null` is always
    permitted so the model can distinguish 'could not determine' from a real
    observation (null != absent)."""
    if field.type == "bool":
        return {"type": ["boolean", "null"]}
    if field.type == "numeric_months":
        return {"type": ["number", "null"]}
    if field.type == "enum":
        assert field.enum is not None
        return {"type": ["string", "null"], "enum": [*field.enum, None]}
    raise ValueError(f"unknown field type {field.type!r} for {field.id}")


def build_json_schema(schema: Schema) -> dict[str, Any]:
    """The output_config.format schema handed to the extraction call.

    Each field is an object {value, evidence_span}. `additionalProperties: false`
    and a full `required` list are mandatory for structured outputs.
    """
    props: dict[str, Any] = {}
    for f in schema.fields:
        props[f.id] = {
            "type": "object",
            "properties": {
                "value": _value_json_type(f),
                "evidence_span": {
                    "type": ["string", "null"],
                    "description": "Verbatim quote from the document supporting the value, or null.",
                },
            },
            "required": ["value", "evidence_span"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": props,
        "required": schema.field_ids,
        "additionalProperties": False,
    }


def coerce_value(field: Field, value: Any) -> Any:
    """Canonicalise a stored value to the field's Python type. Necessary because
    SQLite (and other stores) round-trip booleans as 0/1; without this the
    frequency-key lookup in stats silently misses. None stays None."""
    if value is None:
        return None
    if field.type == "bool":
        return bool(value)
    if field.type == "enum":
        return str(value)
    if field.type == "numeric_months":
        return float(value)
    return value


def coerce_row(schema: Schema, row: dict[str, Any]) -> dict[str, Any]:
    """Coerce every schema field in a row; pass through non-schema keys."""
    out = dict(row)
    for f in schema.fields:
        if f.id in out:
            out[f.id] = coerce_value(f, out[f.id])
    return out


def bucket_numeric(field: Field, value: float) -> str:
    """Map a numeric_months value to a discrete bucket label for co-occurrence."""
    if value >= PERPETUAL_SENTINEL:
        return "perpetual"
    edges = field.bins or [12, 24, 36, 60]
    lo = None
    for edge in edges:
        if value <= edge:
            return f"<={int(edge)}mo" if lo is None else f"{int(lo)+1}-{int(edge)}mo"
        lo = edge
    return f">{int(edges[-1])}mo"


def token(field: Field, value: Any) -> str | None:
    """A discrete `field=bucket` token for stats. Returns None for null values
    (excluded from co-occurrence and marginals)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if field.type == "bool":
        return f"{field.id}={'true' if value else 'false'}"
    if field.type == "enum":
        return f"{field.id}={value}"
    if field.type == "numeric_months":
        return f"{field.id}={bucket_numeric(field, float(value))}"
    return None
