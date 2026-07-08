"""Azure OpenAI extraction backend (gpt-4.1).

Same schema and Extraction contract as every other provider — only transport
differs. Uses the classic Azure endpoint (api-key header + api-version query).

Key from AZURE_OPENAI_KEY. Never written to disk.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from ..extract import Extraction, _to_extraction
from ..schema_loader import Schema, load_schema
from .hyperbolic import SYSTEM_PROMPT, _extract_json_object, _field_guide

# Portable config via env (no resource specifics or secrets baked into code):
#   AZURE_OPENAI_ENDPOINT  full chat/completions URL incl. ?api-version=...
#   AZURE_OPENAI_KEY       api key
#   AZURE_OPENAI_MODEL     deployment name (optional; default "gpt-4.1")
MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4.1")


def _endpoint() -> str:
    ep = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not ep:
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT to your Azure chat/completions URL, e.g. "
            "https://<resource>.openai.azure.com/openai/deployments/<deployment>/"
            "chat/completions?api-version=2025-01-01-preview"
        )
    return ep


def _call(messages: list[dict], key: str) -> str:
    body = {
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        _endpoint(),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "api-key": key},
    )
    with urllib.request.urlopen(req, timeout=150) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def extract_text(
    document_text: str,
    *,
    source_name: str = "<text>",
    schema: Schema | None = None,
    license_class: str = "unknown",
    key: str | None = None,
) -> Extraction:
    schema = schema or load_schema()
    key = key or os.environ["AZURE_OPENAI_KEY"]
    user = (
        "Schema fields:\n" + _field_guide(schema)
        + "\n\nReturn a JSON object whose keys are EXACTLY these field ids, each "
          "mapping to {\"value\": ..., \"evidence_span\": ...}.\n\n"
        + "=== NDA DOCUMENT ===\n" + document_text
    )
    reply = _call(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        key,
    )
    parsed = _extract_json_object(reply)
    return _to_extraction(parsed, schema, Path(source_name), license_class, model=MODEL)


def extract_file(path: Path | str, **kw) -> Extraction:
    path = Path(path)
    return extract_text(path.read_text(encoding="utf-8", errors="replace"),
                        source_name=path.name, **kw)
