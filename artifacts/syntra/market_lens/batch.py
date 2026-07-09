"""Anthropic Message Batches API — async, ~50% cheaper bulk extraction.

Same schema and Extraction contract as extract.py's synchronous path: the
same pydantic model (schema_loader.build_extraction_model), the same
tri-state Literal encoding, the same _canonicalize_value conversion back to
Python True/False/None. Only the transport differs — requests are queued as
one Anthropic Message Batch, processed asynchronously server-side (usually
under an hour, up to 24h), and results are collected afterward instead of
returned inline.

Use this for bulk re-extraction (e.g. re-running the crosscheck corpus) where
an immediate response isn't needed; use extract.py's extract_text/extract_file
for single-document, interactive extraction.

Workflow (see scripts/claude_crosscheck_batch.py for a full example):

    manifest = submit_batch(docs)                  # docs: list[(doc_id, text)]
    json.dump(manifest, open("batch.json", "w"))   # <-- persist: this can outlive the process
    ...
    manifest = json.load(open("batch.json"))
    wait_for_batch(manifest["batch_id"])            # blocks, polling, until done
    results = collect_batch_results(manifest)       # -> {doc_id: Extraction | None}

`manifest` is a plain, JSON-serializable dict — there's no dedicated
save/load helper here because json.dump/json.load already do that job; the
only reason to persist it is that a real batch can take up to 24h, longer
than you'd want a script to block, so submit and collect are commonly two
separate process invocations.

custom_id constraints (Anthropic): must match ^[a-zA-Z0-9_-]{1,64}$. Corpus
doc ids aren't guaranteed to fit that alphabet or length, so submit_batch
assigns opaque sequential custom_ids and records the doc_id/field mapping in
the manifest; collect_batch_results needs the manifest (not just the raw
batch_id) to reassemble one Extraction per document from its request(s).

UNVERIFIED against a live API call (no ANTHROPIC_API_KEY in the sandbox this
was written in). The request/response shapes and the field-batch merge logic
were exercised against a mocked `client.messages.batches` in a scratch test
alongside this change, but the first real submit_batch/wait_for_batch/
collect_batch_results call against the live API is also the first live test
of this module.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .extract import (
    MAX_TOKENS,
    MODEL,
    SYSTEM_PROMPT,
    FIELD_BATCH_SIZE,
    Extraction,
    _batch_fields,
    _canonicalize_value,
    _to_extraction,
)
from .schema_loader import Field, Schema, build_extraction_model, load_schema

MANIFEST_VERSION = 1


def _new_client(client: Any | None) -> Any:
    if client is not None:
        return client
    import anthropic  # imported lazily so the offline pipeline needs no SDK

    return anthropic.Anthropic()


def _output_config(schema: Schema, fields: list[Field]) -> dict[str, Any]:
    """Same schema Anthropic's `messages.parse` would build for this field
    subset (see extract.py), reused via the SDK's own public transform_schema
    so the batch path and the synchronous path send byte-identical schemas."""
    import anthropic

    model_cls = build_extraction_model(schema, fields=fields)
    return {"format": {"type": "json_schema", "schema": anthropic.transform_schema(model_cls)}}


def submit_batch(
    docs: list[tuple[str, str]],
    *,
    schema: Schema | None = None,
    client: Any | None = None,
    field_batch_size: int = FIELD_BATCH_SIZE,
) -> dict[str, Any]:
    """Submit one Anthropic Message Batch covering every (doc_id, document_text)
    pair in `docs`. Each doc contributes one request per field-batch (normally
    just one — see extract.py's FIELD_BATCH_SIZE note on why the whole schema
    now fits in a single call). Returns the manifest needed by wait_for_batch
    and collect_batch_results; does not persist it — see module docstring."""
    schema = schema or load_schema()
    client = _new_client(client)

    field_batches = _batch_fields(schema.fields, field_batch_size)
    requests: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for doc_idx, (doc_id, text) in enumerate(docs):
        for batch_idx, fields in enumerate(field_batches):
            custom_id = f"r{doc_idx}b{batch_idx}"
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "thinking": {"type": "disabled"},
                    "system": SYSTEM_PROMPT,
                    "output_config": _output_config(schema, fields),
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "text", "text": "Extract the NDA fields per the schema."},
                        ],
                    }],
                },
            })
            entries.append({
                "custom_id": custom_id,
                "doc_id": doc_id,
                "field_ids": [f.id for f in fields],
            })

    batch = client.messages.batches.create(requests=requests)
    return {
        "manifest_version": MANIFEST_VERSION,
        "batch_id": batch.id,
        "schema_version": schema.version,
        "doc_count": len(docs),
        "request_count": len(requests),
        "entries": entries,
    }


def wait_for_batch(
    batch_id: str,
    *,
    client: Any | None = None,
    poll_interval_s: float = 30.0,
    on_poll: Callable[[Any], None] | None = None,
) -> Any:
    """Block until the batch's processing_status is 'ended', polling every
    poll_interval_s. Batches usually finish well under an hour but can take
    up to 24h — this can block a long time by design; run it in a long-lived
    script, not inline in a request handler. `on_poll(batch)` fires after
    every poll (e.g. for a progress log of batch.request_counts)."""
    client = _new_client(client)
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if on_poll is not None:
            on_poll(batch)
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_interval_s)


def collect_batch_results(
    manifest: dict[str, Any],
    *,
    schema: Schema | None = None,
    license_class: str = "unknown",
    client: Any | None = None,
) -> dict[str, Extraction | None]:
    """Stream a completed batch's results and reassemble one Extraction per
    doc_id (merging field-batches back together — see submit_batch).

    A doc_id maps to None only if EVERY field-batch for it failed (errored /
    expired / canceled / refused / failed validation). A doc_id with a
    PARTIAL failure (some field-batches ok, one missing) still returns an
    Extraction built from whichever fields did come back — the gap shows up
    honestly in `coverage` and in `meta['batch_partial_failures']`, never
    silently backfilled.
    """
    schema = schema or load_schema()
    if manifest["schema_version"] != schema.version:
        raise ValueError(
            f"manifest built for schema {manifest['schema_version']}, "
            f"current is {schema.version} — results would misalign"
        )
    client = _new_client(client)

    by_custom_id = {e["custom_id"]: e for e in manifest["entries"]}
    parsed_by_doc: dict[str, dict[str, Any]] = {}
    failures: dict[str, list[str]] = {}

    for line in client.messages.batches.results(manifest["batch_id"]):
        entry = by_custom_id.get(line.custom_id)
        if entry is None:
            continue  # not from this manifest
        doc_id = entry["doc_id"]
        fields = [schema.by_id(fid) for fid in entry["field_ids"]]
        result = line.result

        if result.type == "errored":
            err = result.error.error
            failures.setdefault(doc_id, []).append(
                f"{line.custom_id}: errored ({err.type}: {err.message})")
            continue
        if result.type != "succeeded":
            failures.setdefault(doc_id, []).append(f"{line.custom_id}: {result.type}")
            continue

        message = result.message
        if message.stop_reason == "refusal":
            failures.setdefault(doc_id, []).append(f"{line.custom_id}: refusal")
            continue
        text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            failures.setdefault(doc_id, []).append(f"{line.custom_id}: no text block in response")
            continue

        model_cls = build_extraction_model(schema, fields=fields)
        try:
            validated = model_cls.model_validate_json(text)
        except Exception as e:
            failures.setdefault(doc_id, []).append(f"{line.custom_id}: validation error: {e}")
            continue

        cell_map = validated.model_dump()
        for f in fields:
            cell_map[f.id]["value"] = _canonicalize_value(f, cell_map[f.id]["value"])
        parsed_by_doc.setdefault(doc_id, {}).update(cell_map)

    out: dict[str, Extraction | None] = {}
    for doc_id in dict.fromkeys(e["doc_id"] for e in manifest["entries"]):  # stable order
        parsed = parsed_by_doc.get(doc_id)
        if not parsed:
            out[doc_id] = None
            continue
        ext = _to_extraction(parsed, schema, Path(doc_id), license_class, model=MODEL)
        if doc_id in failures:
            ext.meta["batch_partial_failures"] = failures[doc_id]
        out[doc_id] = ext
    return out
