"""Syntra — AI-native general counsel. Flask entry point."""
import os
import sys
import json
import datetime
import hashlib
import atexit
import threading
import traceback
from pathlib import Path
from functools import wraps
from flask import (
    Flask, session, redirect, url_for, request,
    render_template, flash, send_file, abort, jsonify,
)

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

from database import init_db, seed_from_file, dump_to_file, get_db
from audit import AuditLog
from pipeline.ingestor import Ingestor
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
        out.append(d)
    return out


# ── pipeline runner ───────────────────────────────────────────────────────────
def _run_pipeline(doc_id: str, path: Path, source_type: str, filename: str, user_id: str):
    playbook = load_playbook()
    lane_two = LaneTwoAgent()

    doc = Ingestor().ingest(path.read_bytes(), source_type, doc_id)
    segment = Segmenter().segment(doc)
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
        ))

    redline_bytes = Redliner().redline(doc, v_objects)
    (UPLOADS / f"{doc_id}_redline.docx").write_bytes(redline_bytes)

    queue_items = Router().route(doc_id, v_objects)

    with get_db() as db:
        db.executemany("INSERT OR REPLACE INTO clauses VALUES (?,?,?,?,?,?)", clause_rows)
        db.executemany("INSERT OR REPLACE INTO classifications VALUES (?,?,?,?)", clf_rows)
        db.executemany("INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?,?,?,?)", verdict_rows)
        for qi in queue_items:
            db.execute(
                "INSERT OR REPLACE INTO queue_items VALUES (?,?,?,?,?,?,?)",
                (qi.item_id, qi.doc_id, qi.priority, qi.assignee,
                 qi.status, datetime.datetime.utcnow().isoformat(), None),
            )
        db.execute(
            "UPDATE documents SET status='processed', side=?, service_line=? WHERE doc_id=?",
            (segment.side, segment.service_line, doc_id),
        )
    AuditLog().append_simple(user_id, "pipeline_run", doc_id)


def _run_pipeline_bg(doc_id: str, path: Path, source_type: str, filename: str, user_id: str):
    """Wrapper that runs _run_pipeline in a thread and updates _jobs."""
    try:
        _run_pipeline(doc_id, path, source_type, filename, user_id)
        _jobs[doc_id] = {"status": "done", "error": None}
    except Exception:
        err = traceback.format_exc()
        print(f"[pipeline] ERROR for {doc_id}:\n{err}")
        _jobs[doc_id] = {"status": "error", "error": err}
        with get_db() as db:
            db.execute("UPDATE documents SET status='error' WHERE doc_id=?", (doc_id,))


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    with get_db() as db:
        docs = db.execute("SELECT * FROM documents ORDER BY uploaded_at DESC LIMIT 10").fetchall()
        pending = db.execute("SELECT COUNT(*) FROM queue_items WHERE status='pending'").fetchone()[0]
    return render_template("index.html", docs=docs, pending=pending, user=current_user())


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


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        f = request.files.get("contract")
        if not f or not f.filename:
            flash("No file selected.")
            return render_template("upload.html", user=current_user())
        ext = f.filename.rsplit(".", 1)[-1].lower()
        if ext not in ("docx", "pdf"):
            flash("Only .docx and .pdf files are supported.")
            return render_template("upload.html", user=current_user())
        doc_id = hashlib.md5(
            (f.filename + str(datetime.datetime.utcnow())).encode()
        ).hexdigest()[:12]
        save_path = UPLOADS / f"{doc_id}.{ext}"
        f.save(save_path)
        user = current_user()
        # Insert the document row immediately so the processing page can find it
        with get_db() as db:
            db.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, ext if ext == "pdf" else "docx", "processing",
                 None, None, user["user_id"],
                 datetime.datetime.utcnow().isoformat(), f.filename),
            )
        _jobs[doc_id] = {"status": "running", "error": None}
        t = threading.Thread(
            target=_run_pipeline_bg,
            args=(doc_id, save_path, ext if ext == "pdf" else "docx", f.filename, user["user_id"]),
            daemon=True,
        )
        t.start()
        return redirect(url_for("processing", doc_id=doc_id))
    return render_template("upload.html", user=current_user())


@app.route("/contracts/<doc_id>/processing")
@login_required
def processing(doc_id):
    with get_db() as db:
        doc = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    if not doc:
        abort(404)
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


@app.route("/contracts")
@login_required
def contracts():
    with get_db() as db:
        docs = db.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return render_template("contracts.html", docs=docs, user=current_user())


@app.route("/contracts/<doc_id>")
@login_required
def contract(doc_id):
    with get_db() as db:
        doc = db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not doc:
            abort(404)
        rows = db.execute(
            """SELECT v.*, c.text AS clause_text, cl.clause_type, cl.confidence
               FROM verdicts v
               JOIN clauses c ON v.clause_id=c.clause_id
               JOIN classifications cl ON v.clause_id=cl.clause_id
               WHERE v.doc_id=? ORDER BY v.risk_weight DESC""",
            (doc_id,),
        ).fetchall()
    verdicts = _process_verdicts(rows)
    redline_available = (UPLOADS / f"{doc_id}_redline.docx").exists()
    return render_template("contract.html", doc=doc, verdicts=verdicts,
                           redline_available=redline_available, user=current_user())


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
    editor.save(current_user()["user_id"])
    return redirect(url_for("playbook_view"))


@app.route("/playbook/ai", methods=["POST"])
@login_required
def playbook_ai():
    prompt = (request.json or {}).get("prompt", "")
    if not prompt:
        return {"success": False, "error": "No prompt"}
    editor = PlaybookEditor.load()
    try:
        changes = editor.edit_with_ai(prompt)
        editor.save(current_user()["user_id"])
        return {"success": True, "changes": changes}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.route("/queue")
@login_required
def queue():
    with get_db() as db:
        items = db.execute(
            """SELECT q.*, d.filename, d.service_line, d.side
               FROM queue_items q JOIN documents d ON q.doc_id=d.doc_id
               WHERE q.status='pending' ORDER BY q.priority DESC""",
        ).fetchall()
    return render_template("queue.html", items=items, user=current_user())


@app.route("/queue/<item_id>/review", methods=["GET", "POST"])
@attorney_required
def review(item_id):
    with get_db() as db:
        item = db.execute("SELECT * FROM queue_items WHERE item_id=?", (item_id,)).fetchone()
        if not item:
            abort(404)
        if request.method == "POST":
            decision = request.form.get("decision", "rejected")
            notes = request.form.get("notes", "")
            db.execute(
                "UPDATE queue_items SET status=?, attorney_notes=? WHERE item_id=?",
                (decision, notes, item_id),
            )
            AuditLog().append_simple(
                current_user()["user_id"], f"review_{decision}", item["doc_id"]
            )
            flash(f"Contract {decision}.")
            return redirect(url_for("queue"))
        rows = db.execute(
            """SELECT v.*, c.text AS clause_text, cl.clause_type
               FROM verdicts v
               JOIN clauses c ON v.clause_id=c.clause_id
               JOIN classifications cl ON v.clause_id=cl.clause_id
               WHERE v.doc_id=? ORDER BY v.risk_weight DESC""",
            (item["doc_id"],),
        ).fetchall()
    verdicts = _process_verdicts(rows)
    return render_template("review.html", item=item, verdicts=verdicts, user=current_user())


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
