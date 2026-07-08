"""Party identification — one cheap LLM call at upload time.

Infers the contracting parties so the uploader can confirm which one is
"us" before analysis. The confirmed party sharpens side/service-line
segmentation and, for NDAs, sets the disclosure perspective used by the
side pill and Market Lens favorability. Failure (no key, parse error,
scanned doc) or an explicit skip falls back to the existing automatic
inference — the upload flow must never block on this step.
"""
from __future__ import annotations

import json

# Party names/roles come from the head of the document.
_SAMPLE_CHARS = 4000
_MAX_PARTIES = 4


def infer_parties(text: str) -> list[dict]:
    """Return [{"name": ..., "role": ...}, ...] — [] when unavailable/unclear."""
    from llm import LIGHT_MODEL, audited_chat, llm_available

    if not llm_available() or not (text or "").strip():
        return []

    prompt = (
        "Identify the contracting parties in this contract excerpt.\n"
        "For each party give:\n"
        '- "name": the entity name exactly as written (or the defined alias, '
        'e.g. \'ABC Corp ("Discloser")\').\n'
        '- "role": that party\'s role in THIS contract, e.g. "Disclosing Party", '
        '"Receiving Party", "Both disclose and receive (mutual NDA)", '
        '"Supplier / Service Provider", "Customer / Client", "Licensor", "Licensee".\n'
        "List only actual contracting parties (not affiliates or representatives), "
        f"at most {_MAX_PARTIES}.\n\n"
        f"CONTRACT EXCERPT:\n{text[:_SAMPLE_CHARS]}\n\n"
        'Return JSON: {"parties": [{"name": "...", "role": "..."}]}'
    )
    resp = audited_chat(
        "party_inference", ref=text[:1000],
        model=LIGHT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content or "{}")

    parties: list[dict] = []
    seen: set[str] = set()
    for p in (raw.get("parties") or [])[: _MAX_PARTIES]:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()[:120]
        role = str(p.get("role") or "").strip()[:100]
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        parties.append({"name": name, "role": role})
    return parties


def party_perspective(role: str) -> str:
    """Map a party's contractual role to an NDA perspective."""
    r = (role or "").lower()
    has_disclose = "disclos" in r
    has_receive = "receiv" in r or "recipient" in r
    if "mutual" in r or "both" in r or (has_disclose and has_receive):
        return "mutual"
    if has_disclose:
        return "discloser"
    if has_receive:
        return "recipient"
    return "mutual"
