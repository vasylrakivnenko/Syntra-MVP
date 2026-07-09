"""Load nda-schema.yaml and derive artifacts the pipeline needs:

  - a pydantic model per extraction call, for Anthropic's `messages.parse`
    structured-output path (enforces null != absent -- see build_extraction_model)
  - discrete-token helpers for co-occurrence stats (bool/enum/numeric bucketing)
  - field metadata lookups (type, region, favorability, source)

One schema, three consumers (extract / build_table / stats), so the schema is
loaded here once and nowhere else guesses field types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field as PydanticField, create_model

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


# Sentinel written on the wire for bool/enum fields whose value can't be
# determined. Collapsing null-vs-absent into one flat 3-way Literal (instead
# of a nullable boolean) turns "is it null OR false" from a two-step judgment
# call into a single choice, and -- as a side effect -- a Literal[...] compiles
# to a plain string enum (no anyOf/null union), which matters for Anthropic's
# structured-outputs 16-union-typed-parameter cap (see build_extraction_model).
UNDETERMINED = "undetermined"


def _field_submodel(field: Field) -> type[BaseModel]:
    """Pydantic model for one field's {value, evidence_span} object.

    bool/enum fields use a flat Literal (true/false/undetermined, or the enum
    values plus "undetermined") rather than a nullable type — see UNDETERMINED.
    numeric_months stays a plain nullable float: there's no "absent" state for
    a duration distinct from "couldn't determine it", so the ambiguity that
    motivates the flat Literal for bool/enum doesn't apply here.
    """
    if field.type == "bool":
        value_type: Any = Literal["true", "false", UNDETERMINED]
        value_desc = (
            f"{field.description} Respond 'true' if the clause is present, 'false' if it is "
            "explicitly ABSENT from the document, or 'undetermined' if the text does not let "
            "you tell either way. Never guess between false and undetermined."
        )
    elif field.type == "enum":
        assert field.enum is not None
        value_type = Literal[tuple(field.enum) + (UNDETERMINED,)]
        value_desc = (
            f"{field.description} Choose exactly one of {list(field.enum)}, or 'undetermined' "
            "if the document does not let you determine this."
        )
    elif field.type == "numeric_months":
        value_type = Optional[float]
        value_desc = (
            f"{field.description} Normalize to months (2 years -> 24). Perpetual/indefinite -> "
            f"{PERPETUAL_SENTINEL}. null if it cannot be determined from the text."
        )
    else:
        raise ValueError(f"unknown field type {field.type!r} for {field.id}")

    return create_model(
        f"Field_{field.id}",
        __config__=ConfigDict(extra="forbid"),
        value=(value_type, PydanticField(description=value_desc)),
        evidence_span=(
            str,
            PydanticField(description="Verbatim quote from the document supporting the value, "
                                       "or empty string if value is false/undetermined/null."),
        ),
    )


def build_extraction_model(schema: Schema, fields: list[Field] | None = None) -> type[BaseModel]:
    """The pydantic model passed as `output_format` to `client.messages.parse`.

    One nested submodel per field (see _field_submodel), each carrying the
    field's own extraction guidance as its `value` description — previously
    this guidance (the YAML `description` text) was never sent to the model at
    all on this path, which is a materially bigger source of ambiguity than
    the null-vs-false question the Literal collapsing addresses.

    `fields`: restrict the model to a subset (for batched calls); defaults to
    every field in `schema`.
    """
    target = fields if fields is not None else schema.fields
    kwargs = {f.id: (_field_submodel(f), ...) for f in target}
    return create_model("NDAExtraction", __config__=ConfigDict(extra="forbid"), **kwargs)


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
