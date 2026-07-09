"""Syntra — AI-native general counsel. Flask entry point."""
import os
import re
import sys
import json
import datetime
import hashlib
import atexit
import threading
import traceback
from pathlib import Path
from functools import wraps
from collections import Counter
from flask import (
    Flask, session, redirect, url_for, request,
    render_template, flash, send_file, abort, jsonify,
)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from database import init_db, seed_from_file, dump_to_file, get_db
from audit import AuditLog
from pipeline.ingestor import Ingestor
from pipeline.parties import infer_parties, party_perspective
from pipeline.segmenter import Segmenter
from pipeline.chunker import Chunker
from pipeline.classifier import Classifier
from pipeline.triage import Triage
from pipeline.redliner import Redliner
from pipeline.router import Router
from knowledge.lane_two import LaneTwoAgent
from playbook.editor import PlaybookEditor

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("SESSION_SECRET", "syntra-dev-secret-change-me")

UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)

# ── startup / shutdown ────────────────────────────────────────────────────────
init_db()
seed_from_file()
# Pipeline jobs live in an in-memory dict, so a restart strands any doc left in
# 'processing'. Mark them failed so they can be re-uploaded (dedupe skips errors).
with get_db() as _db:
    _db.execute("UPDATE documents SET status='error' WHERE status='processing'")
atexit.register(dump_to_file)

# ── background job tracker ─────────────────────────────────────────────────────
# Maps doc_id → {"status": "running"|"done"|"error", "error": str|None}
_jobs: dict[str, dict] = {}


# ── auth helpers ──────────────────────────────────────────────────────────────
def current_user():
    return session.get("user")


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def attorney_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        u = current_user()
        if not u or u["role"] != "attorney":
            abort(403)
        return f(*args, **kwargs)
    return wrapper


def load_playbook():
    return PlaybookEditor.load().playbook


def _process_verdicts(rows):
    """Parse JSON fields in DB verdict rows."""
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["rule_ids"] = json.loads(d.get("rule_ids") or "[]")
        except Exception:
            d["rule_ids"] = []
        try:
            raw = d.get("cited_position")
            d["cited_position"] = json.loads(raw) if raw else None
        except Exception:
            d["cited_position"] = None
        out.append(d)
    return _enrich_citations(out)


def _enrich_citations(verdicts):
    """Rows analyzed before citation snapshots existed get a best-effort
    citation from the CURRENT playbook, labeled as_of='current' so templates
    can distinguish it from an at-analysis snapshot."""
    if all(d.get("cited_position") or not d.get("rule_ids") for d in verdicts):
        return verdicts
    playbook = load_playbook()
    positions = {}  # rule_id -> (service_line, policy_id, position)
    for sl in playbook.service_lines:
        for pol_id, pos in sl.positions.items():
            positions.setdefault(pos.id, (sl, pol_id, pos))
    policies = {p.id: p for p in playbook.policies}
    for d in verdicts:
        if d.get("cited_position") or not d.get("rule_ids"):
            continue
        hit = positions.get(d["rule_ids"][0])
        if not hit:
            continue
        sl, pol_id, pos = hit
        policy = policies.get(pol_id)
        d["cited_position"] = {
            "rule_id": pos.id,
            "policy_id": pol_id,
            "policy_label": policy.label if policy else pol_id,
            "clause_type": policy.clause_type if policy else d.get("clause_type"),
            "service_line": sl.id,
            "preferred": pos.preferred,
            "fallback": pos.fallback,
            "walk_away": pos.walk_away,
            "playbook_version": playbook.version,
            "as_of": "current",
        }
    return verdicts


# ── contract review view builder — presentation layer ONLY ───────────────────
# Internal enums (unacceptable / acceptable_deviation / complies / unusual /
# silence / abstain) never change; they are mapped to plain-language display
# categories at render time. No pipeline or data-model impact.

_RULE_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")
_PREAMBLE_RE = re.compile(r"^\s*EX-\d|\.htm\b|^\s*UNITED STATES\s+SECURITIES", re.I)
_CLAUSE_NO_RE = re.compile(r"^\s*(\d{1,2})[.)]\s")

_CLAUSE_TITLES = {
    "term_and_termination": "Term & Termination",
    "limitation_of_liability": "Limitation of Liability",
    "ip_ownership": "IP Ownership",
    "other": "General Provision",
}

# display category -> (label, badge class, left-border accent class, plural noun)
_CATEGORIES = {
    "needs_change":   ("Needs change", "cat-badge-danger", "accent-danger",
                       "clauses that need changes"),
    "compromise":     ("Acceptable compromise", "cat-badge-warning", "accent-warning",
                       "acceptable compromises"),
    "couldnt_assess": ("Couldn't assess", "cat-badge-muted", "",
                       "clauses we couldn't assess"),
    "standard":       ("Standard", "cat-badge-success", "",
                       "standard clauses"),
    "not_covered":    ("Not covered", "cat-badge-muted", "",
                       "clauses your playbook doesn't cover"),
}
_GROUP_ORDER = ["needs_change", "compromise", "couldnt_assess", "standard", "not_covered"]


def _display_category(v):
    if v.get("branch") == "silence":
        return "not_covered"
    if v.get("branch") == "abstain":
        return "couldnt_assess"
    return {
        "unacceptable": "needs_change",
        "acceptable_deviation": "compromise",
        "complies": "standard",
    }.get(v.get("status"), "couldnt_assess")  # 'unusual' = couldn't confirm


def _plain_sentence(text, max_words=20):
    """First sentence, playbook IDs stripped, word-capped — P3 microcopy rules."""
    if not text:
        return ""
    t = _RULE_ID_RE.sub("", text)
    t = re.sub(r"\(\s*\)", "", t)            # parens emptied by ID removal
    t = re.sub(r"\s+", " ", t).strip(" \t-–—,;:")
    t = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0].strip()
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(",;:") + "…"
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    if t and t[-1] not in ".…!?":
        t += "."
    return t


def _takeaway(v, cat):
    """One-line collapsed-card summary: 'so what' in <=20 words, no rule IDs."""
    if v.get("branch") == "abstain":
        reason = v.get("reason") or ""
        if reason.startswith("Clause classified"):
            return "Your playbook has no position for this type of clause — not assessed."
        if reason.startswith("Insufficient grounding"):
            return "Not enough information in this clause alone to check it against your playbook."
        if reason.startswith("LLM"):
            return "Automated analysis wasn't available for this clause."
        return _plain_sentence(reason) or "Couldn't assess this clause."
    if v.get("branch") == "silence":
        return _plain_sentence(v.get("reason")) or \
            "Your playbook doesn't cover this clause yet — no position to compare against."
    base = _plain_sentence(v.get("rationale"))
    if cat == "standard":
        return (base + " No action needed.").strip() if base else \
            "Matches your playbook position. No action needed."
    if cat == "couldnt_assess":
        return base or "Couldn't confirm compliance from this clause alone."
    return base or "Review this clause."


def _clause_title(v):
    base = _CLAUSE_TITLES.get(v.get("clause_type")) or \
        (v.get("clause_type") or "Clause").replace("_", " ").title()
    m = _CLAUSE_NO_RE.match(v.get("clause_text") or "")
    return f"{base} — Clause {m.group(1)}" if m else base


def _market_share(field):
    if not field or not field.get("note"):
        return None
    m = re.search(r"(\d+)%", field["note"])
    return int(m.group(1)) if m else None


def _market_rows(market):
    """4-5 scannable label + plain-sentence rows for the Market Lens panel."""
    fields = {f.get("id"): f for f in (market.get("fields") or []) if f.get("determined")}
    rows = []

    def add(label, text, share_field=None):
        share = _market_share(fields.get(share_field)) if share_field else None
        rows.append({"label": label, "text": text, "share": share,
                     "uncommon": share is not None and share < 25})

    f = fields.get("ci_definition_breadth")
    if f:
        text = {
            "marked only": "Only marked information counts — label sensitive documents before sharing.",
            "defined categories": "Defined categories of information are protected.",
            "broad": "Broadly defined — most information you share is protected.",
        }.get(f["value"], str(f["value"]).capitalize() + ".")
        add("What's protected", text, "ci_definition_breadth")

    f = fields.get("confidentiality_survival_months")
    if f:
        text = "Indefinite — the duty to keep secrets never expires." \
            if f["value"] == "perpetual" else \
            f"Obligations last {f['value']} after the agreement ends."
        add("Confidentiality duration", text, "confidentiality_survival_months")

    f = fields.get("term_months")
    if f:
        text = "No fixed term." if f["value"] in ("perpetual", "unspecified") \
            else f"The agreement runs for {f['value']}."
        add("Agreement term", text, "term_months")

    ns, nc = fields.get("non_solicit"), fields.get("non_compete")
    if ns or nc:
        extras = [name for name, x in (("non-solicit", ns), ("non-compete", nc))
                  if x and x.get("value") == "yes"]
        text = "Includes " + " and ".join(extras) + " restrictions — broader than a pure NDA." \
            if extras else "None — keeps the NDA focused on confidentiality."
        add("Non-solicit / non-compete", text)

    dr, inj = fields.get("dispute_resolution"), fields.get("injunctive_relief")
    if dr or inj:
        parts = []
        if inj:
            parts.append("Court injunctions allowed to stop misuse"
                         if inj["value"] == "yes" else "No injunctive-relief clause")
        if dr:
            parts.append("forum unspecified" if dr["value"] == "unspecified"
                         else f"disputes via {dr['value']}")
        add("Dispute resolution", "; ".join(parts) + ".", "dispute_resolution")

    return rows[:5]


def _contract_view(verdicts, market, escalated=False):
    """Everything the redesigned /contracts/<id> template needs, derived
    at render time from existing verdict rows and the stored market report.
    `escalated` = a pending attorney queue item exists; keeps the headline
    tone in lockstep with the routing pill and the dashboard risk chip."""
    cards, preamble = [], []
    first_start = min((v.get("clause_start") or 0 for v in verdicts), default=0)
    for v in verdicts:
        text_head = (v.get("clause_text") or "")[:150]
        if (v.get("clause_start") or 0) == first_start and verdicts and \
                _PREAMBLE_RE.search(text_head):
            preamble.append(v)
            continue
        cat = _display_category(v)
        label, badge, accent, _plural = _CATEGORIES[cat]
        cards.append({**v, "cat": cat, "cat_label": label, "badge": badge,
                      "accent": accent, "title": _clause_title(v),
                      "takeaway": _takeaway(v, cat), "open": False})

    counts = Counter(c["cat"] for c in cards)

    groups = []
    for key in _GROUP_ORDER:
        members = [c for c in cards if c["cat"] == key]
        if key == "needs_change":
            members.sort(key=lambda c: -(c.get("risk_weight") or 0))
        else:
            members.sort(key=lambda c: c.get("clause_start") or 0)
        label, _badge, _accent, plural = _CATEGORIES[key]
        groups.append({"key": key, "label": label, "plural": plural,
                       "count": len(members), "cards": members})

    blocker = fix = None
    if counts["needs_change"]:
        top = next(g for g in groups if g["key"] == "needs_change")["cards"][0]
        top["open"] = True
        cp = top.get("cited_position") or {}
        blocker = {
            "sentence": _plain_sentence(top.get("rationale") or top.get("reason"), 25)
                        or top["takeaway"],
            "rule_id": cp.get("rule_id") or (top.get("rule_ids") or [None])[0],
            "rule_href": (f"/playbook#pos-{cp['service_line']}-{cp['policy_id']}"
                          if cp.get("service_line") and cp.get("policy_id") else None),
            "clause_id": top.get("clause_id"),
        }
        fix_card = top if top.get("suggested_text") else next(
            (c for g in groups if g["key"] == "needs_change"
             for c in g["cards"] if c.get("suggested_text")), None)
        if fix_card:
            fix = {"text": fix_card["suggested_text"], "clause_id": fix_card["clause_id"]}

    if counts["needs_change"]:
        headline, tone = "Changes required before signing", "danger"
    elif counts["compromise"]:
        headline, tone = "OK to sign with minor compromises", "warning"
    elif counts["couldnt_assess"] or counts["not_covered"]:
        headline, tone = "No blockers found — some clauses couldn't be fully checked", "muted"
    elif cards:
        headline, tone = "Ready to sign", "success"
    elif preamble:
        headline, tone = "Only header text found — nothing to analyze", "muted"
    else:
        headline, tone = "No analysis available yet", "muted"
    if escalated and tone in ("success", "muted"):
        # never show a green/neutral verdict next to an attorney-escalation pill
        tone = "warning"

    parts = []
    if counts["compromise"]:
        n = counts["compromise"]
        parts.append(f"{n} clause{'s are' if n != 1 else ' is an'} acceptable compromise{'s' if n != 1 else ''}")
    if counts["couldnt_assess"]:
        n = counts["couldnt_assess"]
        parts.append(f"{n} clause{'s' if n != 1 else ''} need{'' if n != 1 else 's'} info we couldn't verify from this document")
    if counts["not_covered"]:
        n = counts["not_covered"]
        parts.append(f"{n} clause{'s aren' if n != 1 else ' isn'}'t covered by your playbook yet")
    if counts["standard"]:
        parts.append("everything else is standard")
    summary_line = " · ".join(parts)

    return {
        "headline": headline, "tone": tone, "blocker": blocker, "fix": fix,
        "summary_line": summary_line, "counts": counts, "groups": groups,
        "preamble": preamble, "total": len(cards),
        "market_rows": _market_rows(market) if market else [],
    }


# ── pipeline runner ───────────────────────────────────────────────────────────
def _run_pipeline(doc_id: str, path: Path, source_type: str, filename: str, user_id: str,
                  our_party: dict | None = None):
    from llm import set_llm_actor
    set_llm_actor(user_id)  # attribute every model call in this run to the uploader
    playbook = load_playbook()
    lane_two = LaneTwoAgent()

    doc = Ingestor().ingest(path.read_bytes(), source_type, doc_id)
    segment = Segmenter().segment(doc, playbook, our_party=our_party)
    # A confirmed party pins the NDA perspective (mutual/recipient/discloser);
    # it is stored in documents.side for NDA docs and drives the side pill,
    # triage display, and Market Lens favorability.
    if our_party and (segment.service_line or "").startswith("nda"):
        segment.side = party_perspective(our_party.get("role", ""))
    clauses = Chunker().chunk(doc)

    clause_rows, clf_rows, verdict_rows, v_objects = [], [], [], []

    for clause in clauses:
        clf = Classifier().classify(clause)
        improved = lane_two.run(clause, clf)
        if improved:
            clf = improved
        verdict = Triage().decide(clause, clf, segment, playbook)
        v_objects.append(verdict)

        rw = next(
            (p.risk_weight for p in playbook.policies if p.clause_type == clf.clause_type),
            3,
        )
        clause_rows.append((clause.id, doc_id, clause.text, clause.start,
                            clause.end, json.dumps(clause.heading_path)))
        clf_rows.append((clause.id, clf.clause_type, clf.confidence, json.dumps(clf.spans)))
        verdict_rows.append((
            verdict.clause_id, doc_id, verdict.branch, verdict.status,
            json.dumps(verdict.rule_ids), verdict.rationale, verdict.reason,
            verdict.service_line, rw, verdict.suggested_text or "",
            json.dumps(verdict.cited_position) if verdict.cited_position else None,
        ))

    redline_bytes = Redliner().redline(doc, v_objects, clauses)
    (UPLOADS / f"{doc_id}_redline.docx").write_bytes(redline_bytes)

    queue_items = Router().route(doc_id, v_objects)

    with get_db() as db:
        db.executemany("INSERT OR REPLACE INTO clauses VALUES (?,?,?,?,?,?)", clause_rows)
        db.executemany("INSERT OR REPLACE INTO classifications VALUES (?,?,?,?)", clf_rows)
        db.executemany(
            """INSERT OR REPLACE INTO verdicts
               (clause_id, doc_id, branch, status, rule_ids, rationale, reason,
                service_line, risk_weight, suggested_text, cited_position)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            verdict_rows,
        )
        # Reprocessing must not leave stale pending queue items behind.
        db.execute("DELETE FROM queue_items WHERE doc_id=? AND status='pending'", (doc_id,))
        for qi in queue_items:
            db.execute(
                """INSERT OR REPLACE INTO queue_items
                   (item_id, doc_id, priority, assignee, status, reason,
                    created_at, attorney_notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (qi.item_id, qi.doc_id, qi.priority, qi.assignee,
                 qi.status, qi.reason,
                 datetime.datetime.utcnow().isoformat(), None),
            )
        db.execute(
            "UPDATE documents SET status='processed', side=?, service_line=? WHERE doc_id=?",
            (segment.side, segment.service_line, doc_id),
        )
    AuditLog().append_simple(user_id, "pipeline_run", doc_id)

    # Market benchmarking for NDAs. Raw off-market stats stay advisory; routing
    # reacts only when the favorability assessment judges a flagged combination
    # unfavorable to our position AND the playbook missed it. Failures here
    # must never break core analysis.
    if (segment.service_line or "").startswith("nda"):
        try:
            from models import QueueItem
            from pipeline.market import (assess_market_flags, market_escalation_reason,
                                         market_escalations, run_market_lens)
            report = run_market_lens(doc.full_text, filename)
            try:
                clause_types = {r[0]: r[1] for r in clf_rows}
                playbook_findings = [
                    {"clause_type": clause_types.get(v.clause_id, "missing clause"),
                     "status": v.status or v.branch,
                     "rationale": v.rationale or v.reason or ""}
                    for v in v_objects
                    if v.branch == "silence"
                    or (v.branch == "verdict"
                        and v.status in ("unacceptable", "acceptable_deviation"))
                ]
                perspective = (segment.side if segment.side in NDA_PERSPECTIVES
                               else _NDA_PERSPECTIVE.get(segment.service_line, "mutual"))
                report["assessments"] = assess_market_flags(report, perspective,
                                                            playbook_findings)
            except Exception:
                # Assessment is enrichment — keep the advisory card either way.
                report["assessments"] = []
                print(f"[market-lens] assessment failed for {doc_id}:\n{traceback.format_exc()}")
            escalations = market_escalations(report["assessments"])
            with get_db() as db:
                db.execute(
                    "INSERT OR REPLACE INTO market_reports VALUES (?,?,?,?)",
                    (doc_id, report["schema_version"], json.dumps(report),
                     datetime.datetime.utcnow().isoformat()),
                )
                if escalations:
                    reason = market_escalation_reason(escalations)
                    existing = db.execute(
                        """SELECT item_id, reason FROM queue_items
                           WHERE doc_id=? AND status='pending'
                           ORDER BY rowid DESC LIMIT 1""",
                        (doc_id,),
                    ).fetchone()
                    if existing:
                        # Already escalated by the playbook — enrich its reason.
                        if "Market Lens:" not in (existing["reason"] or ""):
                            merged = (f"{existing['reason']}; {reason}"
                                      if existing["reason"] else reason)
                            db.execute("UPDATE queue_items SET reason=? WHERE item_id=?",
                                       (merged, existing["item_id"]))
                    else:
                        # Floor at 3 so an escalated item never shows "Low risk".
                        qi = QueueItem(doc_id=doc_id,
                                       priority=max(3, 2 * len(escalations)),
                                       reason=reason)
                        db.execute(
                            """INSERT OR REPLACE INTO queue_items
                               (item_id, doc_id, priority, assignee, status, reason,
                                created_at, attorney_notes)
                               VALUES (?,?,?,?,?,?,?,?)""",
                            (qi.item_id, qi.doc_id, qi.priority, qi.assignee,
                             qi.status, qi.reason,
                             datetime.datetime.utcnow().isoformat(), None),
                        )
            if escalations:
                # Routing decisions belong in the audit trail.
                AuditLog().append_simple(user_id, "market_lens_escalation", doc_id)
        except Exception:
            print(f"[market-lens] skipped for {doc_id}:\n{traceback.format_exc()}")


def _run_pipeline_bg(doc_id: str, path: Path, source_type: str, filename: str, user_id: str,
                     our_party: dict | None = None):
    """Wrapper that runs _run_pipeline in a thread and updates _jobs."""
    try:
        _run_pipeline(doc_id, path, source_type, filename, user_id, our_party=our_party)
        _jobs[doc_id] = {"status": "done", "error": None}
    except Exception:
        err = traceback.format_exc()
        print(f"[pipeline] ERROR for {doc_id}:\n{err}")
        _jobs[doc_id] = {"status": "error", "error": err}
        with get_db() as db:
            db.execute("UPDATE documents SET status='error' WHERE doc_id=?", (doc_id,))


# NDA rows use NDA-appropriate perspective labels instead of supplier/customer.
# "nda" covers legacy documents segmented before service lines were playbook-bound.
_NDA_PERSPECTIVE = {"nda_standalone": "mutual", "nda": "mutual"}
NDA_PERSPECTIVES = ("mutual", "recipient", "discloser")
app.add_template_global(NDA_PERSPECTIVES, "NDA_PERSPECTIVES")


@app.template_global("side_display")
def side_display(service_line, side):
    # A party-confirmed NDA perspective is stored directly in documents.side.
    if side in NDA_PERSPECTIVES:
        return side
    return _NDA_PERSPECTIVE.get(service_line or "", side or "—")


@app.template_global("nda_perspective")
def nda_perspective(service_line, side=None):
    """Return the active NDA perspective for a service line, or None if not an NDA."""
    if side in NDA_PERSPECTIVES:
        return side
    return _NDA_PERSPECTIVE.get(service_line or "")


@app.template_global("risk_label")
def risk_label(priority):
    """Translate the machine risk score into a human label + bootstrap color."""
    p = priority or 0
    if p >= 5:
        return "High risk", "danger"
    if p >= 3:
        return "Medium risk", "warning text-dark"
    return "Low risk", "secondary"


@app.template_global("days_until")
def days_until(datestr):
    """Days from today until an ISO date; None if unparseable."""
    try:
        return (datetime.date.fromisoformat(datestr) - datetime.date.today()).days
    except Exception:
        return None


# Human labels for service-line enum ids — avoids raw ids like "nda_standalone"
# leaking to the UI as a title-cased mess ("Nda Standalone").
_SERVICE_LINE_LABELS = {
    "nda_standalone": "NDA",
    "nda": "NDA",
    "general_supplier": "Supplier Agreement",
    "general_customer": "Customer Agreement",
}


@app.template_global("service_line_display")
def service_line_display(service_line):
    if not service_line:
        return "—"
    return _SERVICE_LINE_LABELS.get(service_line, service_line.replace("_", " ").title())


@app.template_global("human_date")
def human_date(iso_str):
    """Short human date for table cells; pass the raw iso_str separately for a tooltip."""
    if not iso_str:
        return "—"
    try:
        return datetime.datetime.fromisoformat(iso_str).strftime("%b %-d")
    except Exception:
        return iso_str[:10]


def _attention_sort_key(d):
    """Sort key for 'needs my attention' ordering: pending review / urgent
    float to top, approved/rejected sink — see dashboard UX brief."""
    review_status = d["review_status"]
    urgency = d["urgency"] or "standard"
    resolved = review_status in ("approved", "rejected")
    if resolved:
        tier = 3
    elif review_status == "pending" or urgency == "urgent":
        tier = 0
    elif urgency == "high":
        tier = 1
    else:
        tier = 2
    needed_by = d["needed_by"] or "9999-99-99"
    risk_flags = d["risk_flags"] if "risk_flags" in d.keys() else 0
    return (tier, needed_by, -(risk_flags or 0))


def _sort_docs(docs, sort_mode):
    if sort_mode == "recent":
        return docs
    return sorted(docs, key=_attention_sort_key)


@app.context_processor
def inject_mvp_notice():
    """One-shot MVP disclaimer popup — armed by the upload flow, consumed by
    the next rendered page. Routes may also force it via show_mvp_modal=True."""
    return {"show_mvp_modal": session.pop("mvp_notice", False)}


def _inbox_payload(user):
    """Navbar bell payload: role-aware attention list.

    Attorneys see contracts waiting on their review; everyone else sees
    attorney decisions on their own uploads that they have not yet
    acknowledged (clicking the bell notification acknowledges — a plain
    page view/refresh must NOT clear the bell).
    """
    if not user:
        return None
    items = []
    try:
        with get_db() as db:
            if user["role"] == "attorney":
                count = db.execute(
                    "SELECT COUNT(*) AS n FROM queue_items WHERE status='pending'",
                ).fetchone()["n"]
                rows = db.execute(
                    """SELECT q.item_id, q.doc_id, q.reason, q.priority,
                              d.filename, d.urgency, d.needed_by
                       FROM queue_items q JOIN documents d ON q.doc_id=d.doc_id
                       WHERE q.status='pending'
                       ORDER BY CASE d.urgency WHEN 'urgent' THEN 2 WHEN 'high' THEN 1 ELSE 0 END DESC,
                                CASE WHEN d.needed_by IS NULL THEN 1 ELSE 0 END,
                                d.needed_by ASC,
                                q.priority DESC LIMIT 8""",
                ).fetchall()
                items = [{
                    "url": url_for("review", item_id=r["item_id"]),
                    "filename": r["filename"],
                    "label": ("urgent" if r["urgency"] == "urgent"
                              else "high" if r["urgency"] == "high"
                              else risk_label(r["priority"])[0]),
                    "badge": ("danger" if r["urgency"] == "urgent"
                              else "warning text-dark" if r["urgency"] == "high"
                              else risk_label(r["priority"])[1]),
                    "detail": (f"Due {r['needed_by']} · " if r["needed_by"] else "")
                              + (r["reason"] or "Escalated for review")[:80],
                } for r in rows]
                title = "Awaiting your review"
            else:
                count = db.execute(
                    """SELECT COUNT(*) AS n
                       FROM queue_items q JOIN documents d ON q.doc_id=d.doc_id
                       WHERE q.status IN ('approved','rejected')
                         AND q.acknowledged_at IS NULL AND d.uploaded_by=?""",
                    (user["user_id"],),
                ).fetchone()["n"]
                rows = db.execute(
                    """SELECT q.doc_id, q.status, q.reviewed_at, q.reviewed_by,
                              q.attorney_notes, d.filename
                       FROM queue_items q JOIN documents d ON q.doc_id=d.doc_id
                       WHERE q.status IN ('approved','rejected')
                         AND q.acknowledged_at IS NULL AND d.uploaded_by=?
                       ORDER BY q.reviewed_at DESC LIMIT 8""",
                    (user["user_id"],),
                ).fetchall()
                items = [{
                    "url": url_for("contract", doc_id=r["doc_id"], ack=1),
                    "filename": r["filename"],
                    "label": r["status"],
                    "badge": "success" if r["status"] == "approved" else "danger",
                    "detail": (f"by {r['reviewed_by']}" if r["reviewed_by"] else "by attorney")
                              + (f" · {r['reviewed_at'][:10]}" if r["reviewed_at"] else ""),
                } for r in rows]
                title = "Attorney decisions"
    except Exception:
        return None
    return {"count": count, "entries": items, "title": title}


@app.context_processor
def inject_inbox():
    return {"inbox": _inbox_payload(current_user())}


@app.route("/notifications.json")
@login_required
def api_notifications():
    """Live poll target for the navbar bell (role-aware).

    NOTE: must NOT live under /api/* — the workspace proxy routes that
    prefix to a different artifact, so the Flask app never sees it."""
    return jsonify(_inbox_payload(current_user()))


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200


_DOCS_QUERY = """
    SELECT d.*,
           (SELECT q.status FROM queue_items q
             WHERE q.doc_id = d.doc_id ORDER BY q.rowid DESC LIMIT 1) AS review_status,
           (SELECT COUNT(*) FROM verdicts v
             WHERE v.doc_id = d.doc_id AND v.status = 'unacceptable') AS risk_unacceptable,
           (SELECT COUNT(*) FROM verdicts v
             WHERE v.doc_id = d.doc_id
               AND ((v.branch = 'verdict' AND v.status != 'complies') OR v.branch = 'silence')
           ) AS risk_flags
    FROM documents d
    WHERE d.version = (SELECT MAX(d2.version) FROM documents d2
                        WHERE d2.case_id = d.case_id)
"""


@app.route("/")
@login_required
def index():
    sort_mode = "recent" if request.args.get("sort") == "recent" else "attention"
    user = current_user()
    with get_db() as db:
        docs = db.execute(_DOCS_QUERY + " ORDER BY d.uploaded_at DESC").fetchall()
        pending = db.execute("SELECT COUNT(*) FROM queue_items WHERE status='pending'").fetchone()[0]
        week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat()
        this_week = db.execute(
            "SELECT COUNT(*) FROM documents WHERE uploaded_at >= ?", (week_ago,)
        ).fetchone()[0]
    playbook = load_playbook()
    rule_count = sum(len(sl.positions) for sl in playbook.service_lines)
    docs = _sort_docs(docs, sort_mode)[:10]
    return render_template(
        "index.html", docs=docs, pending=pending, user=user, sort_mode=sort_mode,
        this_week=this_week, rule_count=rule_count,
        playbook_version=playbook.version.split(".")[0],
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        role = request.form.get("role", "owner")
        if not username:
            flash("Username is required.")
            return render_template("login.html")
        uid = hashlib.md5(username.encode()).hexdigest()[:12]
        with get_db() as db:
            db.execute(
                "INSERT OR IGNORE INTO users VALUES (?,?,?,?)",
                (uid, username, role, datetime.datetime.utcnow().isoformat()),
            )
            user = db.execute("SELECT * FROM users WHERE user_id=?", (uid,)).fetchone()
        session["user"] = dict(user)
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _eligible_cases():
    """Latest version per case that has finished analysis — the valid targets
    for a new-version upload. In-flight cases (processing/awaiting_party) are
    excluded so a running pipeline is never raced."""
    with get_db() as db:
        return db.execute(
            """SELECT d.doc_id, d.case_id, d.version, d.filename, d.uploaded_at,
                      d.status,
                      (SELECT q.status FROM queue_items q
                        WHERE q.doc_id = d.doc_id AND q.status != 'superseded'
                        ORDER BY q.rowid DESC LIMIT 1) AS review_status
               FROM documents d
               WHERE d.status IN ('processed', 'error')
                 AND d.version = (SELECT MAX(d2.version) FROM documents d2
                                   WHERE d2.case_id = d.case_id)
               ORDER BY d.uploaded_at DESC"""
        ).fetchall()


def _escalate_urgency(doc_row, urgency, needed_by):
    """Merge urgency/deadline onto an existing doc — escalate, never
    downgrade. Returns True if anything changed."""
    rank = {"standard": 0, "high": 1, "urgent": 2}
    new_urgency = (urgency if rank.get(urgency, 0) > rank.get(doc_row["urgency"] or "standard", 0)
                   else doc_row["urgency"])
    new_deadline = (min(filter(None, (needed_by, doc_row["needed_by"])))
                    if (needed_by or doc_row["needed_by"]) else None)
    if (new_urgency, new_deadline) == (doc_row["urgency"], doc_row["needed_by"]):
        return False
    with get_db() as db:
        db.execute("UPDATE documents SET urgency=?, needed_by=? WHERE doc_id=?",
                   (new_urgency, new_deadline, doc_row["doc_id"]))
    return True


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        def form_again():
            return render_template(
                "upload.html", user=current_user(), cases=_eligible_cases(),
                preselect_case=request.form.get("parent_case_id", ""))
        f = request.files.get("contract")
        if not f or not f.filename:
            flash("No file selected.")
            return form_again()
        ext = f.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("docx", "pdf"):
            flash("Only .docx and .pdf files are supported.")
            return form_again()
        urgency = request.form.get("urgency", "standard")
        if urgency not in ("standard", "high", "urgent"):
            urgency = "standard"
        needed_by = request.form.get("needed_by", "").strip() or None
        if needed_by:
            try:
                datetime.date.fromisoformat(needed_by)
            except ValueError:
                needed_by = None
        file_bytes = f.read()
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        mode = request.form.get("mode", "new")
        parent = None
        if mode == "version":
            # ── new version of an existing case ──
            parent_case_id = request.form.get("parent_case_id", "").strip()
            with get_db() as db:
                parent = db.execute(
                    """SELECT * FROM documents
                       WHERE case_id=? AND version = (SELECT MAX(version) FROM documents
                                                       WHERE case_id=?)""",
                    (parent_case_id, parent_case_id),
                ).fetchone()
            # Server-side eligibility check — the dropdown can be stale.
            if not parent or parent["status"] not in ("processed", "error"):
                flash("Pick which contract this file is a new version of — the case "
                      "must have finished its previous analysis first.")
                return form_again()
            if parent["content_hash"] == content_hash:
                # Identical to the case's current version — no new version;
                # reuse the escalate-never-downgrade urgency semantics.
                if _escalate_urgency(parent, urgency, needed_by):
                    flash(f"This file is identical to the current version (v{parent['version']}) "
                          "— updated its urgency/deadline instead of creating a new version.")
                else:
                    flash(f"This file is identical to the current version (v{parent['version']}) "
                          "— showing the existing analysis.")
                session["mvp_notice"] = True
                return redirect(url_for("contract", doc_id=parent["doc_id"]))
        else:
            # ── new contract: global same-file dedupe. Re-uploading the same
            # file as a NEW contract stays the intentional escape hatch to
            # escalate urgency on an existing one — never downgrade. ──
            with get_db() as db:
                # Only match each case's LATEST version — redirecting into a
                # superseded page (hidden from every list) would strand the user.
                dup = db.execute(
                    """SELECT doc_id, status, urgency, needed_by FROM documents
                       WHERE content_hash=? AND status IN ('processing','processed','awaiting_party')
                         AND version = (SELECT MAX(version) FROM documents d2
                                        WHERE d2.case_id = documents.case_id)
                       ORDER BY uploaded_at DESC LIMIT 1""",
                    (content_hash,),
                ).fetchone()
            if dup:
                if _escalate_urgency(dup, urgency, needed_by):
                    flash("This exact file has already been analysed — updated its "
                          "urgency/deadline and showing the existing analysis.")
                else:
                    flash("This exact file has already been analysed — showing the existing analysis.")
                session["mvp_notice"] = True
                target = {"processing": "processing",
                          "awaiting_party": "select_party"}.get(dup["status"], "contract")
                return redirect(url_for(target, doc_id=dup["doc_id"]))
        doc_id = hashlib.md5(
            (f.filename + str(datetime.datetime.utcnow())).encode()
        ).hexdigest()[:12]
        save_path = UPLOADS / f"{doc_id}.{ext}"
        save_path.write_bytes(file_bytes)
        user = current_user()
        source_type = ext if ext == "pdf" else "docx"
        # Infer the contracting parties (one cheap LLM call) so the uploader
        # can confirm which one is "us" before analysis. Any failure falls
        # back to the fully automatic flow — upload never blocks on this.
        parties: list[dict] = []
        try:
            from llm import set_llm_actor
            set_llm_actor(user["user_id"])
            text = Ingestor().ingest(file_bytes, source_type, doc_id).full_text
            parties = infer_parties(text)
        except Exception:
            print(f"[parties] inference failed for {doc_id}:\n{traceback.format_exc()}")
        awaiting = len(parties) >= 2
        now = datetime.datetime.utcnow().isoformat()
        # Insert the document row immediately so the next page can find it
        with get_db() as db:
            if parent is not None:
                # Version number is assigned inside the INSERT so two
                # concurrent uploads to the same case can't collide.
                db.execute(
                    """INSERT OR REPLACE INTO documents
                       (doc_id, source_type, status, side, service_line,
                        uploaded_by, uploaded_at, filename, content_hash, parties_json,
                        urgency, needed_by, case_id, version)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,
                               (SELECT COALESCE(MAX(version),0)+1 FROM documents WHERE case_id=?))""",
                    (doc_id, source_type,
                     "awaiting_party" if awaiting else "processing",
                     None, None, user["user_id"], now, f.filename, content_hash,
                     json.dumps(parties) if parties else None,
                     urgency, needed_by, parent["case_id"], parent["case_id"]),
                )
                # Retire stale attorney work: prior versions' pending reviews
                # vanish from the queue; the new analysis re-routes on its own.
                db.execute(
                    """UPDATE queue_items SET status='superseded'
                       WHERE status='pending'
                         AND doc_id IN (SELECT doc_id FROM documents
                                         WHERE case_id=? AND doc_id != ?)""",
                    (parent["case_id"], doc_id),
                )
                # Uploading a revision answers the decision — clear the
                # operator's unacknowledged-decision bell for this case.
                db.execute(
                    """UPDATE queue_items SET acknowledged_at=?
                       WHERE acknowledged_at IS NULL AND status IN ('approved','rejected')
                         AND doc_id IN (SELECT doc_id FROM documents WHERE case_id=?)""",
                    (now, parent["case_id"]),
                )
            else:
                db.execute(
                    """INSERT OR REPLACE INTO documents
                       (doc_id, source_type, status, side, service_line,
                        uploaded_by, uploaded_at, filename, content_hash, parties_json,
                        urgency, needed_by, case_id, version)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                    (doc_id, source_type,
                     "awaiting_party" if awaiting else "processing",
                     None, None, user["user_id"], now, f.filename, content_hash,
                     json.dumps(parties) if parties else None,
                     urgency, needed_by, doc_id),
                )
        if parent is not None:
            try:  # audit AFTER the write block commits (SQLite single-writer)
                AuditLog().append_simple(
                    user["user_id"], "version_uploaded",
                    f"{parent['case_id']}:{doc_id}",
                )
            except Exception:
                pass  # never block an upload on audit enrichment
        # Arm the one-shot MVP disclaimer popup for the next page render.
        session["mvp_notice"] = True
        if awaiting:
            return redirect(url_for("select_party", doc_id=doc_id))
        _jobs[doc_id] = {"status": "running", "error": None}
        t = threading.Thread(
            target=_run_pipeline_bg,
            args=(doc_id, save_path, source_type, f.filename, user["user_id"]),
            daemon=True,
        )
        t.start()
        return redirect(url_for("processing", doc_id=doc_id))
    return render_template("upload.html", user=current_user(),
                           cases=_eligible_cases(),
                           preselect_case=request.args.get("case", ""))


@app.route("/contracts/<doc_id>/select-party", methods=["GET", "POST"])
@login_required
def select_party(doc_id):
    """Confirm which contracting party is 'us' before the pipeline runs."""
    with get_db() as db:
        doc = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if not doc:
        abort(404)
    if doc["status"] != "awaiting_party":
        target = "processing" if doc["status"] == "processing" else "contract"
        return redirect(url_for(target, doc_id=doc_id))
    try:
        parties = json.loads(doc["parties_json"] or "[]")
    except Exception:
        parties = []
    if request.method == "POST":
        choice = request.form.get("party", "skip")
        our_party = None
        if choice.isdigit() and int(choice) < len(parties):
            our_party = parties[int(choice)]
        user = current_user()
        with get_db() as db:
            # Atomic flip guards against a double-POST starting two pipelines.
            cur = db.execute(
                """UPDATE documents SET status='processing', our_party=?
                   WHERE doc_id=? AND status='awaiting_party'""",
                (json.dumps(our_party) if our_party else None, doc_id),
            )
            if cur.rowcount != 1:
                return redirect(url_for("processing", doc_id=doc_id))
        AuditLog().append_simple(
            user["user_id"],
            "party_confirmed" if our_party else "party_skipped",
            doc_id,
        )
        path = UPLOADS / f"{doc_id}.{doc['source_type']}"
        _jobs[doc_id] = {"status": "running", "error": None}
        t = threading.Thread(
            target=_run_pipeline_bg,
            args=(doc_id, path, doc["source_type"], doc["filename"],
                  user["user_id"], our_party),
            daemon=True,
        )
        t.start()
        return redirect(url_for("processing", doc_id=doc_id))
    return render_template("select_party.html", doc=doc, parties=parties,
                           user=current_user())


@app.route("/contracts/<doc_id>/processing")
@login_required
def processing(doc_id):
    with get_db() as db:
        doc = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if not doc:
        abort(404)
    if doc["status"] == "awaiting_party":
        return redirect(url_for("select_party", doc_id=doc_id))
    job = _jobs.get(doc_id, {"status": "done", "error": None})
    if job["status"] == "done":
        return redirect(url_for("contract", doc_id=doc_id))
    if job["status"] == "error":
        flash(f"Pipeline error: {job['error'][:300] if job['error'] else 'Unknown error'}")
        return redirect(url_for("upload"))
    return render_template("processing.html", doc=doc, doc_id=doc_id, user=current_user())


@app.route("/contracts/<doc_id>/status")
@login_required
def job_status(doc_id):
    job = _jobs.get(doc_id, {"status": "done", "error": None})
    return jsonify(job)


@app.route("/contracts/<doc_id>/review-status")
@login_required
def review_status(doc_id):
    """Live poll target: latest attorney review status for a document."""
    with get_db() as db:
        row = db.execute(
            "SELECT status FROM queue_items WHERE doc_id=? ORDER BY rowid DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
    return jsonify({"review_status": row["status"] if row else None})


@app.route("/contracts")
@login_required
def contracts():
    sort_mode = "recent" if request.args.get("sort") == "recent" else "attention"
    with get_db() as db:
        docs = db.execute(_DOCS_QUERY + " ORDER BY d.uploaded_at DESC").fetchall()
    docs = _sort_docs(docs, sort_mode)
    return render_template("contracts.html", docs=docs, user=current_user(), sort_mode=sort_mode)


def _issue_counter(db, doc_id):
    """Multiset of a doc's open findings, keyed by human-readable label — the
    unit of comparison for the version-history changes summary."""
    rows = db.execute(
        """SELECT v.branch, v.status, v.reason, cl.clause_type
           FROM verdicts v
           LEFT JOIN classifications cl ON v.clause_id = cl.clause_id
           WHERE v.doc_id=?
             AND (v.branch='silence' OR (v.branch='verdict' AND v.status != 'complies'))""",
        (doc_id,),
    ).fetchall()
    counts = Counter()
    for r in rows:
        if r["branch"] == "silence":
            label = f"Missing: {(r['reason'] or 'required clause').strip()}"
        else:
            ctype = (r["clause_type"] or "clause").replace("_", " ").title()
            status = (r["status"] or "").replace("_", " ")
            label = f"{ctype} — {status}"
        counts[label] += 1
    return counts


def _version_changes(db, versions):
    """Analysis-comparison summary between consecutive processed versions:
    {doc_id: {new: [labels], resolved: [labels], still_open: int}}. This is a
    comparison of pipeline findings, not a text diff."""
    changes = {}
    if len(versions) < 2:
        return changes
    prev = None
    for v in versions:
        if v["status"] != "processed":
            continue  # diff against the nearest prior *processed* version
        cur = _issue_counter(db, v["doc_id"])
        if prev is not None:
            new, resolved = cur - prev, prev - cur
            changes[v["doc_id"]] = {
                "new": sorted(new.elements()),
                "resolved": sorted(resolved.elements()),
                "still_open": sum((cur & prev).values()),
            }
        prev = cur
    return changes


@app.route("/contracts/<doc_id>")
@login_required
def contract(doc_id):
    with get_db() as db:
        doc = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not doc:
            abort(404)
        if doc["status"] == "awaiting_party":
            return redirect(url_for("select_party", doc_id=doc_id))
        # Clicking the bell notification (?ack=1) acknowledges the attorney
        # decision. A plain page view/refresh must NOT clear the bell — the
        # operator may see the status banner without having noticed the bell.
        acked = 0
        if doc["uploaded_by"] == current_user()["user_id"] and request.args.get("ack"):
            acked = db.execute(
                """UPDATE queue_items SET acknowledged_at=?
                   WHERE doc_id=? AND status IN ('approved','rejected')
                     AND acknowledged_at IS NULL""",
                (datetime.datetime.utcnow().isoformat(), doc_id),
            ).rowcount
        rows = db.execute(
            """SELECT v.*, c.text AS clause_text, c.start AS clause_start,
                      cl.clause_type, cl.confidence
               FROM verdicts v
               JOIN clauses c ON v.clause_id=c.clause_id
               JOIN classifications cl ON v.clause_id=cl.clause_id
               WHERE v.doc_id=? ORDER BY v.risk_weight DESC""",
            (doc_id,),
        ).fetchall()
    if acked:
        try:  # read receipt on a legal decision belongs in the audit chain
            AuditLog().append_simple(
                current_user()["user_id"], "decision_acknowledged", doc_id
            )
        except Exception:
            pass  # never block the contract page on audit enrichment
    verdicts = _process_verdicts(rows)
    with get_db() as db:
        queue_item = db.execute(
            "SELECT * FROM queue_items WHERE doc_id=? ORDER BY rowid DESC LIMIT 1", (doc_id,)
        ).fetchone()
        mr = db.execute(
            "SELECT report_json FROM market_reports WHERE doc_id=?", (doc_id,)
        ).fetchone()
        versions = db.execute(
            """SELECT d.doc_id, d.version, d.filename, d.uploaded_at, d.status,
                      d.urgency, d.needed_by, u.username AS uploader,
                      (SELECT q.status FROM queue_items q
                        WHERE q.doc_id = d.doc_id AND q.status != 'superseded'
                        ORDER BY q.rowid DESC LIMIT 1) AS review_status
               FROM documents d LEFT JOIN users u ON u.user_id = d.uploaded_by
               WHERE d.case_id=? ORDER BY d.version ASC""",
            (doc["case_id"],),
        ).fetchall()
        version_changes = _version_changes(db, versions)
    market = None
    if mr:
        try:
            market = json.loads(mr["report_json"])
        except Exception:
            market = None
    our_party = None
    if doc["our_party"]:
        try:
            our_party = json.loads(doc["our_party"])
        except Exception:
            our_party = None
    redline_available = (UPLOADS / f"{doc_id}_redline.docx").exists()
    latest = versions[-1] if versions else None
    is_latest = (latest is None) or (latest["doc_id"] == doc["doc_id"])
    view = _contract_view(verdicts, market,
                          escalated=bool(queue_item and queue_item["status"] == "pending"))
    return render_template("contract.html", doc=doc, verdicts=verdicts, queue_item=queue_item,
                           market=market, our_party=our_party, view=view,
                           redline_available=redline_available, user=current_user(),
                           versions=versions, version_changes=version_changes,
                           latest=latest, is_latest=is_latest)


@app.route("/contracts/<doc_id>/redline-preview")
@login_required
def redline_preview(doc_id):
    with get_db() as db:
        doc = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not doc:
            abort(404)
        rows = db.execute(
            """SELECT v.*, c.text AS clause_text, cl.clause_type
               FROM verdicts v
               JOIN clauses c ON v.clause_id=c.clause_id
               JOIN classifications cl ON v.clause_id=cl.clause_id
               WHERE v.doc_id=? ORDER BY c.start ASC""",
            (doc_id,),
        ).fetchall()
    verdicts = _process_verdicts(rows)
    return render_template("redline_preview.html", doc=doc, verdicts=verdicts,
                           user=current_user(), show_mvp_modal=True)


@app.route("/contracts/<doc_id>/download")
@login_required
def download_redline(doc_id):
    path = UPLOADS / f"{doc_id}_redline.docx"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"redline_{doc_id}.docx")


@app.route("/playbook")
@login_required
def playbook_view():
    pb = load_playbook()
    return render_template("playbook.html", playbook=pb, user=current_user())


@app.route("/playbook/save", methods=["POST"])
@login_required
def playbook_save():
    editor = PlaybookEditor.load()
    action = request.form.get("action")
    if action == "edit_cell":
        editor.edit_cell(
            request.form["service_line_id"], request.form["policy_id"],
            {"preferred": request.form.get("preferred", ""),
             "fallback": request.form.get("fallback", ""),
             "walk_away": request.form.get("walk_away", "")},
        )
    elif action == "add_row":
        editor.add_row(request.form["label"], request.form["side"])
    elif action == "delete_row":
        editor.delete_row(request.form["service_line_id"])
    elif action == "add_column":
        editor.add_column(request.form["label"], request.form["clause_type"])
    elif action == "delete_column":
        editor.delete_column(request.form["policy_id"])
    elif action == "rename_row":
        editor.rename_row(request.form["service_line_id"], request.form["new_label"])
    elif action == "rename_column":
        editor.rename_column(request.form["policy_id"], request.form["new_label"])
    user_id = current_user()["user_id"]
    version = editor.save(user_id)
    # Audit AFTER save() returns — its get_db block has committed by then.
    AuditLog().append_simple(
        user_id, f"playbook_{action or 'edit'}",
        json.dumps({"version": version, **request.form.to_dict()}, sort_keys=True),
    )
    return redirect(url_for("playbook_view"))


@app.route("/playbook/ai", methods=["POST"])
@login_required
def playbook_ai():
    prompt = (request.json or {}).get("prompt", "")
    if not prompt:
        return {"success": False, "error": "No prompt"}
    editor = PlaybookEditor.load()
    user_id = current_user()["user_id"]
    try:
        from llm import set_llm_actor
        set_llm_actor(user_id)  # the model call inside edit_with_ai is audited too
        changes = editor.edit_with_ai(prompt)
        version = editor.save(user_id)
        AuditLog().append_simple(
            user_id, "playbook_ai_edit",
            json.dumps({"version": version, "prompt": prompt, "changes": changes},
                       sort_keys=True),
        )
        return {"success": True, "changes": changes}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.route("/queue")
@login_required
def queue():
    with get_db() as db:
        items = db.execute(
            """SELECT q.*, d.filename, d.service_line, d.side, d.urgency, d.needed_by, d.version
               FROM queue_items q JOIN documents d ON q.doc_id=d.doc_id
               WHERE q.status='pending'
               ORDER BY CASE d.urgency WHEN 'urgent' THEN 2 WHEN 'high' THEN 1 ELSE 0 END DESC,
                        CASE WHEN d.needed_by IS NULL THEN 1 ELSE 0 END,
                        d.needed_by ASC,
                        q.priority DESC""",
        ).fetchall()
    return render_template("queue.html", items=items, user=current_user())


@app.route("/queue/<item_id>/review", methods=["GET", "POST"])
@attorney_required
def review(item_id):
    if request.method == "POST":
        decision = request.form.get("decision", "rejected")
        if decision not in ("approved", "rejected"):
            decision = "rejected"
        notes = request.form.get("notes", "")
        with get_db() as db:
            item = db.execute(
                "SELECT * FROM queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            if not item:
                abort(404)
            # Guard against a stale review page: only a still-pending item may
            # be decided. If a new version superseded it mid-review, this
            # UPDATE matches nothing and we must NOT resurrect the dead item
            # (acknowledged_at=NULL would re-light the operator's bell for a
            # version that no longer matters).
            cur = db.execute(
                """UPDATE queue_items
                   SET status=?, attorney_notes=?, reviewed_at=?, reviewed_by=?,
                       acknowledged_at=NULL
                   WHERE item_id=? AND status='pending'""",
                (decision, notes, datetime.datetime.utcnow().isoformat(),
                 current_user()["username"], item_id),
            )
            decided = cur.rowcount > 0
        if not decided:
            flash("This review was superseded by a newer version of the contract "
                  "— your decision was not recorded. The new version re-routes on its own.")
            return redirect(url_for("queue"))
        # Audit outside the transaction: AuditLog opens its own connection,
        # and SQLite allows only one writer at a time.
        AuditLog().append_simple(
            current_user()["user_id"], f"review_{decision}", item["doc_id"]
        )
        flash(f"Contract {decision}.")
        return redirect(url_for("queue"))
    with get_db() as db:
        item = db.execute(
            """SELECT q.*, d.filename, d.urgency, d.needed_by, d.version, d.case_id
               FROM queue_items q JOIN documents d ON q.doc_id = d.doc_id
               WHERE q.item_id=?""",
            (item_id,),
        ).fetchone()
        if not item:
            abort(404)
        rows = db.execute(
            """SELECT v.*, c.text AS clause_text, cl.clause_type
               FROM verdicts v
               JOIN clauses c ON v.clause_id=c.clause_id
               JOIN classifications cl ON v.clause_id=cl.clause_id
               WHERE v.doc_id=? ORDER BY v.risk_weight DESC""",
            (item["doc_id"],),
        ).fetchall()
        # Context for re-reviews: the most recent decision on a prior version.
        prior_review = None
        if (item["version"] or 1) > 1:
            prior_review = db.execute(
                """SELECT q.status, q.attorney_notes, q.reviewed_by, q.reviewed_at,
                          d.version, d.doc_id
                   FROM queue_items q JOIN documents d ON d.doc_id = q.doc_id
                   WHERE d.case_id=? AND d.version < ?
                     AND q.status IN ('approved','rejected')
                   ORDER BY d.version DESC, q.rowid DESC LIMIT 1""",
                (item["case_id"], item["version"]),
            ).fetchone()
    verdicts = _process_verdicts(rows)
    return render_template("review.html", item=item, verdicts=verdicts,
                           prior_review=prior_review, user=current_user())


@app.route("/queue/<item_id>/promote", methods=["POST"])
@attorney_required
def promote_to_fallback(item_id):
    clause_id = request.form.get("clause_id", "")
    with get_db() as db:
        item = db.execute("SELECT * FROM queue_items WHERE item_id=?", (item_id,)).fetchone()
        clause = db.execute("SELECT * FROM clauses WHERE clause_id=?", (clause_id,)).fetchone()
        clf = db.execute("SELECT * FROM classifications WHERE clause_id=?", (clause_id,)).fetchone()
        verdict = db.execute("SELECT * FROM verdicts WHERE clause_id=?", (clause_id,)).fetchone()
    if item and clause and clf and verdict:
        editor = PlaybookEditor.load()
        editor.promote_fallback(verdict["service_line"] or "", clf["clause_type"], clause["text"])
        editor.save(current_user()["user_id"])
        AuditLog().append_simple(current_user()["user_id"], "promote_to_fallback", clause_id)
        flash("Promoted to fallback in playbook.")
    return redirect(url_for("review", item_id=item_id))


@app.route("/audit")
@login_required
def audit_view():
    with get_db() as db:
        events = db.execute(
            "SELECT * FROM audit_events ORDER BY ts DESC LIMIT 100"
        ).fetchall()
    return render_template("audit.html", events=events, user=current_user())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
