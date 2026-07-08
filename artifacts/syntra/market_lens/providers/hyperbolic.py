"""Hyperbolic extraction backend (OpenAI-compatible gateway to open models).

Same schema and same Extraction contract as the Anthropic path in extract.py —
only the transport differs. Hyperbolic serves open models (Llama/DeepSeek/Qwen),
so this uses the OpenAI-style /v1/chat/completions endpoint with JSON-object
output + the schema embedded in the prompt, then parses and coerces.

Key comes from HYPERBOLIC_API_KEY. Never written to disk.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from ..extract import Extraction, _to_extraction  # reuse the row/coverage builder
from ..schema_loader import Schema, load_schema

ENDPOINT = "https://api.hyperbolic.xyz/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# Cloudflare in front of Hyperbolic blocks non-browser user-agents (403 code 1010).
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

SYSTEM_PROMPT = (
    "You are a precise contract-analysis engine. Extract typed fields from a "
    "Non-Disclosure Agreement into a fixed schema and return ONLY a JSON object. "
    "For every field return an object {\"value\": ..., \"evidence_span\": ...}. "
    "CRITICAL: value=false means the clause is genuinely ABSENT; value=null means "
    "you COULD NOT DETERMINE it from the text. Never guess — if unsure use null, not false. "
    "evidence_span is a short VERBATIM quote supporting the value, or null. "
    "numeric_months: normalize durations to months (2 years -> 24); perpetual/indefinite -> 9999. "
    "enum fields: choose exactly one allowed value or null."
)


def _field_guide(schema: Schema) -> str:
    lines = []
    for f in schema.fields:
        spec = f"- {f.id} ({f.type}"
        if f.type == "enum":
            spec += f"; one of {f.enum}"
        spec += f"): {f.description}"
        lines.append(spec)
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the model's reply into a dict, tolerating stray prose/code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the first balanced {...} block
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model reply")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON object in model reply")


def _call(messages: list[dict], model: str, key: str, *, debug: bool = False) -> str:
    """Streamed chat completion. Streaming is REQUIRED here — non-streaming
    requests sit idle during generation and the gateway kills them at ~130s
    with a 500. SSE keeps the connection alive."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0,
        "stream": True,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}", "User-Agent": _UA},
    )
    parts: list[str] = []
    raw_sample: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for bline in r:
            line = bline.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if debug and len(raw_sample) < 3:
                raw_sample.append(payload)
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
                parts.append(delta.get("content") or "")
            except Exception:
                pass
    text = "".join(parts)
    if debug and not text.strip():
        raise RuntimeError("empty completion; first chunks: " + " || ".join(raw_sample))
    return text


def extract_text(
    document_text: str,
    *,
    source_name: str = "<text>",
    schema: Schema | None = None,
    model: str = MODEL,
    license_class: str = "unknown",
    key: str | None = None,
) -> Extraction:
    schema = schema or load_schema()
    key = key or os.environ["HYPERBOLIC_API_KEY"]

    user = (
        "Schema fields:\n" + _field_guide(schema)
        + "\n\nReturn a JSON object whose keys are EXACTLY these field ids, each "
          "mapping to {\"value\": ..., \"evidence_span\": ...}.\n\n"
        + "=== NDA DOCUMENT ===\n" + document_text
    )
    reply = _call(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        model, key, debug=True,
    )
    parsed = _extract_json_object(reply)
    return _to_extraction(parsed, schema, Path(source_name), license_class, model=model)


def extract_file(path: Path | str, **kw) -> Extraction:
    path = Path(path)
    return extract_text(path.read_text(encoding="utf-8", errors="replace"),
                        source_name=path.name, **kw)
