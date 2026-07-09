"""Market Lens v2 adapter — the one bridge between Syntra and the vendored
market_lens library (see market_lens/ and market_data/).

Three responsibilities, nothing else:
  1. extract_market_row  — whole-doc NDA extraction via Syntra's existing
     OpenAI-compatible client (llm.py), reusing market_lens's own prompt,
     schema and Extraction contract. No new API keys, no anthropic SDK.
  2. score_market_row    — the v2 evidence bundle (market_lens.evidence.
     build_evidence) against the shipped 1,158-NDA market table, offline,
     plus the optional TabPFN signal (pipeline/market_tabpfn.py — skipped
     entirely unless TABPFN_TOKEN is set).
  3. synthesize_market_report — spec.md's "step 3": ONE LLM call that turns
     extraction + evidence bundle + both rarity signals into the
     human-facing Market Position Report. Deliberately not in the library.

Two rarity signals, never collapsed (spec.md §6): rule_univariate (share of
comparable NDAs with this value — transparent frequency count) and
tabpfn_p_obs (learned conditional probability, optional). When they
disagree, that disagreement is surfaced as "worth a second look", not
resolved by picking a winner in code.

Routing policy (unchanged from v1): raw statistical rarity NEVER routes.
The only path to the attorney queue is assess_market_flags(): a cheap-model
judgment that a flagged combination is unfavorable to OUR position and not
already covered by the playbook analysis (see market_escalations()).

Flagging note: the v1.1 calibrated statistic (persisted omx_reference.json)
returns per-combination p-values under a clause-independence null; the
library no longer ships a per-combo "off_market" boolean, so THIS adapter
defines the flag: pvalue <= P_FLAG (0.05). Legacy-shape contributions
(no reference present) are never flagged — that path also logs loudly,
because it means market_data/omx_reference.json went missing.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from llm import LIGHT_MODEL, MODEL, audited_chat
from market_lens.evidence import _load_reference, build_evidence
from market_lens.score import score_against_reference
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
from pipeline.market_tabpfn import tabpfn_signal

MARKET_DIR = Path(__file__).resolve().parent.parent / "market_data"
MARKET_TABLE = MARKET_DIR / "market.sqlite"

# Extraction can time out on very long documents; cap the text we send.
_MAX_DOC_CHARS = 60_000

# A combination is "flagged" (surfaced + eligible for favorability judgment)
# when its co-occurrence is this unlikely under clause independence.
P_FLAG = 0.05

# Both-signals disagreement: one signal calls the value rare, the other
# doesn't. Presented as "worth a second look", never auto-resolved.
_RARE = 0.15

_FAVORABILITY_VALUES = ("favorable", "unfavorable", "neutral", "unclear")


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
    resp = audited_chat(
        "market_extraction", ref=source_name,
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


# ── scoring (offline evidence bundle + optional TabPFN) ──────────────────────

def _table_n() -> int:
    con = sqlite3.connect(MARKET_TABLE)
    try:
        return con.execute("SELECT COUNT(*) FROM ndas").fetchone()[0]
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


def _combo_label(combo: list[str] | tuple[str, ...]) -> str:
    parts = []
    for tok in combo:
        fid, _, bucket = tok.partition("=")
        parts.append(f"{fid.replace('_', ' ')} = {bucket.replace('_', ' ')}")
    return " + ".join(parts)


def _signals_disagree(share: float | None, p_obs: float | None) -> bool:
    """True when exactly one of the two independent signals calls the value
    rare — spec.md §6: surface the disagreement, don't resolve it."""
    if share is None or p_obs is None:
        return False
    return (share < _RARE) != (p_obs < _RARE)


def _field_note(field, entry: dict[str, Any]) -> str | None:
    """One friendly sentence placing the doc's value in the segment, derived
    from the bundle's rule_univariate share."""
    ru = entry.get("rule_univariate")
    if ru is None:
        return None
    share = ru.get("share")
    if share is None:
        return None
    pct = round(share * 100)
    if field.type == "numeric_months":
        return f"{pct}% of comparable NDAs fall in the same range"
    return f"matches {pct}% of comparable NDAs"


def _full_contributions(schema: Schema, row: dict[str, Any],
                        bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """The COMPLETE top-k combo contribution list for this doc.

    build_evidence's per-field rule_combo keeps only the RAREST combo per
    field (right for per-field attribution, lossy as a list — a distinct
    combo whose fields are all claimed by rarer combos would vanish, and
    flagged combos feed favorability → routing). So the list of record is
    re-derived from the persisted reference, exactly as market_lens.score.
    render does; the bundle's per-field view is only a display aid."""
    ref = _load_reference(MARKET_DIR)
    if ref is not None and bundle.get("off_market_status") == "scored":
        om = score_against_reference(schema, row, ref)
        if om is not None:
            out = []
            for c in om.contributions:
                if len(c) == 4:  # v1.1 pvalue statistic
                    combo, obs, exp, pval = c
                    out.append({"combo": list(combo), "observed": obs,
                                "expected": exp, "pvalue": pval})
                else:  # deficit statistic — no calibrated p-value
                    combo, obs, exp = c
                    out.append({"combo": list(combo), "observed": obs, "expected": exp})
            return out
    # Legacy/insufficient-data: fall back to the bundle's (rarest-per-field)
    # view — nothing on this path is ever flagged, so lossiness is harmless.
    out, seen = [], set()
    for entry in (bundle.get("fields") or {}).values():
        c = entry.get("rule_combo")
        if not c:
            continue
        key = tuple(c.get("combo") or ())
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _normalize_contributions(raw: list[dict[str, Any]], seg_n: int) -> list[dict[str, Any]]:
    """Adapter-owned flagging + template-compatible shape (v1.1 pvalue
    statistic preferred; contributions without a p-value never flagged)."""
    out = []
    for c in raw:
        key = tuple(c.get("combo") or ())
        if not key:
            continue
        observed = c.get("observed")
        expected = c.get("expected")
        pvalue = c.get("pvalue")
        out.append({
            "label": _combo_label(key),
            # Field ids behind the combo — used to cite the document evidence
            # spans (report["evidence"]) that grounded each flagged combination.
            "fields": [tok.partition("=")[0] for tok in key],
            "count": observed,
            "n": seg_n,
            "share": round(100 * observed / seg_n, 1) if observed is not None and seg_n else None,
            "observed": observed,
            "expected": round(expected, 1) if isinstance(expected, (int, float)) else expected,
            "pvalue": pvalue,
            "off_market": (pvalue is not None and pvalue <= P_FLAG),
        })
    out.sort(key=lambda c: (c["pvalue"] is None, c["pvalue"] if c["pvalue"] is not None else 1.0))
    return out


def score_market_row(row: dict[str, Any], meta: dict[str, Any] | None = None,
                     schema: Schema | None = None) -> dict[str, Any]:
    """Compare one typed row against the shipped market table via the v2
    evidence bundle. Offline except the optional TabPFN signal."""
    schema = schema or load_schema()
    row = coerce_row(schema, row)

    if not (MARKET_DIR / "omx_reference.json").exists():
        # Silent-downgrade guard: without the persisted reference the library
        # falls back to the legacy uncalibrated index and nothing gets flagged.
        print("[market-lens] WARNING: market_data/omx_reference.json missing — "
              "legacy scoring, no combinations will be flagged")

    tabpfn = tabpfn_signal(schema, row, MARKET_DIR)  # None unless TABPFN_TOKEN set
    bundle = build_evidence(schema, row, MARKET_DIR, tabpfn_p_obs=tabpfn)

    fields = []
    for f in schema.fields:
        entry = bundle["fields"].get(f.id, {})
        value = row.get(f.id)
        ru = entry.get("rule_univariate") or {}
        p_obs = entry.get("tabpfn_p_obs")
        fields.append({
            "id": f.id,
            "label": f.id.replace("_", " "),
            "region": f.region,
            "value": _fmt_value(f, value),
            "determined": value is not None,
            "note": _field_note(f, entry),
            "share": ru.get("share"),
            "tabpfn_p_obs": p_obs,
            "signals_disagree": _signals_disagree(ru.get("share"), p_obs),
        })

    contributions = _normalize_contributions(
        _full_contributions(schema, row, bundle), bundle["segment_n"])

    return {
        "schema_version": schema.version,
        "coverage": (meta or {}).get("coverage"),
        "model": (meta or {}).get("model"),
        "segment": {
            "mutual": row.get("mutual"),
            "n": bundle["segment_n"],
            "table_n": _table_n(),
        },
        "fields": fields,
        "off_market": {
            "index": bundle["off_market_index"],
            "status": bundle["off_market_status"],
            "contributions": contributions,
            "flagged": [c for c in contributions if c["off_market"]],
        },
        "tabpfn_used": tabpfn is not None,
        "bundle": bundle,
    }


# ── step 3: the Market Position Report (spec.md §3/§6 — OUR code) ───────────

_SYNTH_SYSTEM = """You are a senior commercial lawyer writing a short "Market Position Report"
about one NDA, for two audiences at once.

You receive, per extracted field: the value, a verbatim evidence quote from the
document (when available), and up to two independent rarity signals versus a
population of comparable real NDAs:
- rule_share: the share of comparable NDAs with this same value (transparent
  frequency count; LOW = rare).
- tabpfn_p: a learned model's probability of this value given the document's
  other terms (LOW = surprising). May be null if the signal wasn't run.
Where both are present and marked "signals_disagree", explicitly call that
clause out as worth a second look BECAUSE the two signals disagree — do not
silently pick one signal as the truth.

You also receive the whole-document Off-Market Index (0-100; higher = rarer
overall vs comparable NDAs) and the statistically flagged clause combinations.

Respond with JSON only:
{
 "headline": "<one sentence: overall market position of this NDA>",
 "client_summary": "<3-6 plain-language sentences for a business reader: what
   stands out vs the market and why it matters commercially. NO statistics, NO
   percentages, NO model names, NO legal advice — describe, don't advise.>",
 "lawyer_notes": ["<2-6 bullets for the reviewing attorney: the clauses most
   worth attention, each grounded in the numbers (shares/probabilities), noting
   signal disagreement where marked>"]
}

Everything you write is statistical context versus a market sample, not legal
advice — never present rarity as "bad" on its own."""


def synthesize_market_report(ext: Extraction, report: dict[str, Any],
                             source_name: str = "<doc>") -> dict[str, Any]:
    """One LLM call: extraction + evidence bundle -> the human-facing Market
    Position Report. Raises on failure; run_market_lens treats it as optional
    enrichment."""
    bundle = report["bundle"]
    lines = []
    for f in report["fields"]:
        if not f["determined"]:
            continue
        bits = [f"{f['label']}: {f['value']}"]
        if f["share"] is not None:
            bits.append(f"rule_share={f['share']:.2f}")
        if f["tabpfn_p_obs"] is not None:
            bits.append(f"tabpfn_p={f['tabpfn_p_obs']:.2f}")
        if f["signals_disagree"]:
            bits.append("signals_disagree=true")
        quote = (ext.evidence or {}).get(f["id"])
        if quote:
            bits.append(f'quote="{quote[:200]}"')
        lines.append("- " + " | ".join(bits))
    flagged_txt = "\n".join(
        f"- {c['label']} (observed {c['observed']} of {c['n']}; "
        f"~{c['expected']} expected if independent; p={c['pvalue']})"
        for c in report["off_market"]["flagged"]
    ) or "(none)"
    user = (
        f"Segment: {bundle['segment']} (N={bundle['segment_n']} comparable NDAs)\n"
        f"Off-Market Index: {bundle['off_market_index']} "
        f"(status: {bundle['off_market_status']})\n\n"
        f"Fields:\n" + "\n".join(lines)
        + f"\n\nStatistically flagged combinations:\n{flagged_txt}"
    )
    resp = audited_chat(
        "market_synthesis", ref=source_name,
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    return {
        "headline": str(parsed.get("headline", ""))[:300],
        "client_summary": str(parsed.get("client_summary", ""))[:2000],
        "lawyer_notes": [str(n)[:500] for n in parsed.get("lawyer_notes", [])
                         if isinstance(n, str)][:8],
    }


# ── favorability assessment (cheap model, one batched call) ─────────────────

_ASSESS_SYSTEM = """You are a senior commercial lawyer advising "our company" on an NDA.
You will receive: (1) our perspective in this NDA, (2) issues our internal playbook
analysis already raised, and (3) statistically unusual (off-market) clause
combinations found in the NDA — each with how often it was observed among
comparable NDAs versus how often it would be expected if the clauses were
independent (a low p-value means the combination is rarer than chance).

For EACH numbered combination, judge:
- favorability: "favorable" if the unusual terms work in our interest,
  "unfavorable" if they increase our risk or burden, "neutral" if they cut both
  ways or are immaterial, "unclear" if you cannot tell from the information given.
- covered_by_playbook: true if the playbook findings already flag substantially
  the same issue (same clause and same concern), else false.
- rationale: one short sentence.

Judge conservatively: only mark "unfavorable" when the combination plausibly harms
our position. Statistical rarity alone is NOT unfavorable.

Respond with JSON only:
{"assessments": [{"index": 1, "favorability": "...", "covered_by_playbook": true, "rationale": "..."}]}"""


def _combo_stat_txt(c: dict[str, Any]) -> str:
    if c.get("pvalue") is not None:
        return (f"observed in {c['observed']} of {c['n']} comparable NDAs, "
                f"~{c['expected']} expected if independent, p={c['pvalue']}")
    return f"seen in {c.get('count', '?')} of {c.get('n', '?')} comparable NDAs"


def assess_market_flags(report: dict[str, Any], perspective: str,
                        playbook_findings: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Judge each flagged off-market combination from our side's point of view.

    One batched call to the cheap model — returns one assessment per entry in
    report["off_market"]["flagged"], each with favorability, covered_by_playbook
    and a one-line rationale. Raises on failure; callers treat assessments as
    optional enrichment."""
    flagged = report.get("off_market", {}).get("flagged", [])
    if not flagged:
        return []
    findings_txt = "\n".join(
        f"- {f['clause_type']}: {f['status']} — {f['rationale']}".rstrip(" —")
        for f in playbook_findings
    ) or "(none — the playbook analysis raised no issues)"
    combos_txt = "\n".join(
        f"{i + 1}. {c['label']} ({_combo_stat_txt(c)})"
        for i, c in enumerate(flagged)
    )
    user = (
        f"Our perspective: {perspective} — mutual = both parties exchange confidential "
        "information; recipient = we mainly receive it; discloser = we mainly disclose it.\n\n"
        f"Playbook findings already raised:\n{findings_txt}\n\n"
        f"Off-market clause combinations:\n{combos_txt}"
    )
    resp = audited_chat(
        "market_assessment", ref=perspective,
        model=LIGHT_MODEL,
        messages=[
            {"role": "system", "content": _ASSESS_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    by_index = {a.get("index"): a for a in parsed.get("assessments", [])
                if isinstance(a, dict)}
    out = []
    for i, c in enumerate(flagged):
        a = by_index.get(i + 1, {})
        fav = a.get("favorability")
        out.append({
            "label": c["label"],
            "fields": c.get("fields", []),
            "favorability": fav if fav in _FAVORABILITY_VALUES else "unclear",
            "covered_by_playbook": bool(a.get("covered_by_playbook", False)),
            "rationale": str(a.get("rationale", ""))[:300],
            # carry the statistics so the attorney view can ground each
            # judgment in the numbers (old stored reports simply lack these)
            "observed": c.get("observed"),
            "expected": c.get("expected"),
            "pvalue": c.get("pvalue"),
            "n": c.get("n"),
        })
    return out


def market_escalations(assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The only combinations allowed to influence routing: judged against our
    position AND not already covered by the playbook analysis."""
    return [a for a in assessments
            if a["favorability"] == "unfavorable" and not a["covered_by_playbook"]]


def market_escalation_reason(escalations: list[dict[str, Any]]) -> str:
    n = len(escalations)
    shown = "; ".join(a["label"] for a in escalations[:2])
    more = f" (+{n - 2} more)" if n > 2 else ""
    return (f"Market Lens: {n} off-market term combination{'s' if n != 1 else ''} "
            f"judged unfavorable to our position and not covered by the playbook "
            f"— {shown}{more}")


# ── composition ───────────────────────────────────────────────────────────────

def run_market_lens(text: str, source_name: str = "<doc>") -> dict[str, Any]:
    """Extract + score + synthesize one NDA. Extraction/scoring raise on
    failure — callers decide whether market context is allowed to fail softly
    (in the pipeline it is). Synthesis is optional enrichment and fails soft
    HERE (the statistical card must render even if the narrative call dies)."""
    schema = load_schema()
    ext = extract_market_row(text, source_name, schema)
    report = score_market_row(ext.row, ext.meta, schema)
    report["evidence"] = {k: v for k, v in ext.evidence.items() if v}
    try:
        report["synthesis"] = synthesize_market_report(ext, report, source_name)
    except Exception:
        import traceback
        report["synthesis"] = None
        print(f"[market-lens] synthesis failed for {source_name}:\n{traceback.format_exc()}")
    return report
