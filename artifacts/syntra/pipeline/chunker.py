"""Chunker — Document → list[Clause].

Splitting priority:
  0. LLM anchor segmentation   — the model reads the whole document and returns a
     short verbatim anchor (first few words) for each top-level clause; we locate
     those anchors in the text and split there. Robust against nested lists that
     defeat pure regex. Only the anchors are returned by the model, so output is
     cheap regardless of document length.
  1. MarkdownHeaderTextSplitter — for DOCX files with heading styles
  2. Numbered-section splitter   — regex sequence of "1. / 2. / 3." top-level headers
  3. Paragraph-boundary splitter — groups elements into clauses by blank-line gaps
  4. RecursiveCharacterTextSplitter — last resort for dense unstructured text
"""
from __future__ import annotations
import json
import re
from models import Document, Clause

# Matches the start of a numbered legal clause, e.g.:
#   "1. Confidentiality"   "3.1 Payment Terms"   "(a) Something"
_SECTION_START = re.compile(
    r"^\s*"
    r"(?:"
    r"\d{1,2}(?:\.\d{1,2})*\."   # 1.  /  1.1.  /  2.3.
    r"|[A-Z]{1,5}\."              # A.  /  XIV.
    r"|\([a-z]\)"                  # (a)
    r"|\([ivx]+\)"                 # (iv)
    r")"
    r"\s+[A-Z(\"']",
    re.VERBOSE,
)

_MIN_CLAUSE_CHARS = 40   # ignore chunks shorter than this

# 1:1 character canonicalisation used before whitespace-collapsing anchor search
_CANON = {
    "\u201c": '"', "\u201d": '"',           # curly double quotes
    "\u2018": "'", "\u2019": "'", "`": "'",  # curly single quotes
    "\u2014": "-", "\u2013": "-",            # em / en dash
    "\u00a0": " ",                            # non-breaking space
}

_SEGMENT_SYSTEM = (
    "You are a legal contract structure analyzer. Given the full text of a "
    "contract, you identify the boundaries of each top-level clause/section."
)

_SEGMENT_PROMPT = """Analyze the contract below and identify every TOP-LEVEL clause or section, in document order.

Top-level clauses include:
- The preamble / recitals (parties, "WHEREAS" clauses) — treat the entire preamble as ONE clause.
- Each numbered or titled section (e.g. "1. Confidentiality", "2. Term", "Indemnification").
- The signature / execution block, if present.

Do NOT treat nested sub-parts as separate clauses. Items like "(a)", "(b)", "i.", "ii.", or an
inline numbered list inside a section ("shall not include: 1. ... 2. ...") belong to their PARENT
section and must NOT produce their own anchor.

For EACH top-level clause output a short ANCHOR: copy the FIRST 6-10 words of that clause EXACTLY
as they appear in the text — verbatim, same words, punctuation and capitalization. Do not paraphrase
or renumber. Each anchor must be long enough to uniquely locate the start of its clause.

Return STRICT JSON only:
{"anchors": ["<first words of clause 1>", "<first words of clause 2>", ...]}

CONTRACT TEXT:
\"\"\"
%s
\"\"\""""


class Chunker:
    def chunk(self, doc: Document) -> list[Clause]:
        # ── 0. LLM anchor segmentation (primary) ─────────────────────────────
        llm_clauses = self._split_by_llm_anchors(doc)
        if len(llm_clauses) >= 2:
            return llm_clauses

        # ── 1. Markdown-heading split (works well for DOCX) ──────────────────
        md_clauses = self._split_by_markdown_headers(doc)
        if len(md_clauses) >= 2:
            return md_clauses

        # ── 2. Numbered-section split (legal PDFs with "1. Clause" structure) ─
        numbered = self._split_by_numbered_sections(doc)
        if len(numbered) >= 2:
            return numbered

        # ── 3. Paragraph-boundary split (blank-line separated paragraphs) ─────
        paragraphs = self._split_by_paragraphs(doc)
        if len(paragraphs) >= 2:
            return paragraphs

        # ── 4. Character-based fallback ───────────────────────────────────────
        return self._split_by_characters(doc)

    # ── Strategy 0: LLM anchors ──────────────────────────────────────────────

    def _split_by_llm_anchors(self, doc: Document) -> list[Clause]:
        try:
            from llm import get_client, MODEL, llm_available
        except ImportError:
            return []

        text = doc.full_text
        if not llm_available() or len(text) < 200:
            return []

        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SEGMENT_SYSTEM},
                    {"role": "user", "content": _SEGMENT_PROMPT % text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            anchors = data.get("anchors", []) or []
        except Exception as exc:
            print(f"[chunker] LLM anchor segmentation failed: {exc}")
            return []

        if not isinstance(anchors, list) or len(anchors) < 2:
            return []

        # Locate every anchor in a whitespace-normalized copy of the text, then
        # map the hit back to an offset in the ORIGINAL text.
        norm_text, index_map = self._normalize_with_map(text)

        positions: list[int] = []
        search_from = 0
        for anchor in anchors:
            if not isinstance(anchor, str):
                continue
            norm_anchor = self._normalize_str(anchor)
            if len(norm_anchor) < 4:
                continue

            # Search forward from the previous hit only. This preserves document
            # order and avoids latching onto an earlier duplicate of a repeated
            # clause opening (e.g. "Company shall...") which would otherwise
            # create a spurious boundary and cascade to later anchors.
            idx = norm_text.find(norm_anchor, search_from)
            if idx == -1:                                     # loosen: first 4 words
                short = " ".join(norm_anchor.split()[:4])
                idx = norm_text.find(short, search_from) if len(short) >= 12 else -1
            if idx == -1:
                # Unfindable anchor → skip; its clause folds into the previous one.
                continue

            positions.append(index_map[idx])
            search_from = max(search_from, idx + 1)           # never rewind

        positions = sorted(set(positions))
        if len(positions) < 2:
            return []

        # Fold a short leading title (e.g. "NON-DISCLOSURE AGREEMENT") into the
        # first clause; keep a substantial preamble as its own clause.
        if positions[0] > 0:
            if positions[0] <= _MIN_CLAUSE_CHARS:
                positions[0] = 0
            else:
                positions = [0] + positions

        return self._spans_to_clauses(doc, positions)

    # ── Strategy 1: markdown headers ────────────────────────────────────────

    def _split_by_markdown_headers(self, doc: Document) -> list[Clause]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        md_lines: list[str] = []
        for el in doc.elements:
            if el.kind == "heading" and el.heading_path:
                level = min(len(el.heading_path), 3)
                md_lines.append(f"{'#' * level} {el.text}")
            else:
                md_lines.append(el.text)
        md_text = "\n\n".join(md_lines)

        headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split, strip_headers=False
        )
        chunks = splitter.split_text(md_text)
        non_empty = [c for c in chunks if len(c.page_content.strip()) >= _MIN_CLAUSE_CHARS]

        if len(non_empty) <= 1:
            return []

        return self._map_lc_chunks_to_clauses(doc, non_empty)

    # ── Strategy 2: numbered sections ────────────────────────────────────────

    def _split_by_numbered_sections(self, doc: Document) -> list[Clause]:
        """
        Split the full text at top-level numbered section headers ("1. ", "2. " ...).

        Candidate markers are found at line starts, then filtered to the run that
        increments 1, 2, 3, ... — this isolates real top-level sections from nested
        sub-lists (which restart their own numbering) and stray page numbers.
        """
        text = doc.full_text
        marker_re = re.compile(r'(?m)^[ \t]{0,4}(\d{1,2})\.[ \t]+(?=[A-Z_"\u201c])')
        candidates = [(m.start(), int(m.group(1))) for m in marker_re.finditer(text)]
        if len(candidates) < 3:
            return []

        boundaries: list[int] = []
        expected = 1
        for pos, num in candidates:
            if num == expected:
                boundaries.append(pos)
                expected += 1
        if len(boundaries) < 3:
            return []

        if boundaries[0] > _MIN_CLAUSE_CHARS:
            boundaries = [0] + boundaries

        return self._spans_to_clauses(doc, boundaries)

    # ── Strategy 3: paragraph boundaries ─────────────────────────────────────

    def _split_by_paragraphs(self, doc: Document) -> list[Clause]:
        """
        Treat each Document element as a paragraph-level clause.
        Merge very short ones (< 120 chars) with the previous clause.
        """
        merged: list[list[str]] = []
        for el in doc.elements:
            t = el.text.strip()
            if not t:
                continue
            if merged and len(t) < 120:
                merged[-1].append(t)
            else:
                merged.append([t])

        if len(merged) <= 1:
            return []

        return self._groups_to_clauses(doc, merged)

    # ── Strategy 4: character-based ───────────────────────────────────────────

    def _split_by_characters(self, doc: Document) -> list[Clause]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(doc.full_text)
        clauses: list[Clause] = []
        search_start = 0
        for i, t in enumerate(chunks):
            t = t.strip()
            if len(t) < _MIN_CLAUSE_CHARS:
                continue
            anchor = t[:50].strip()
            start = doc.full_text.find(anchor, search_start)
            if start == -1:
                start = search_start
            end = start + len(t)
            search_start = max(search_start, start)
            clauses.append(Clause(
                id=f"{doc.doc_id[:6]}-c{i:03d}", text=t, start=start, end=end,
            ))
        return clauses

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _spans_to_clauses(self, doc: Document, boundaries: list[int]) -> list[Clause]:
        """Turn a sorted list of split offsets into Clause objects with precise spans."""
        text = doc.full_text
        clauses: list[Clause] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            raw = text[start:end]
            lead = len(raw) - len(raw.lstrip())
            chunk = raw.strip()
            if len(chunk) < _MIN_CLAUSE_CHARS:
                continue
            # Precise offsets so full_text[start:end] == chunk (citations may slice it).
            real_start = start + lead
            clauses.append(Clause(
                id=f"{doc.doc_id[:6]}-c{i:03d}",
                text=chunk,
                start=real_start,
                end=real_start + len(chunk),
            ))
        return clauses

    def _groups_to_clauses(self, doc: Document, groups: list[list[str]]) -> list[Clause]:
        clauses: list[Clause] = []
        search_start = 0
        for i, lines in enumerate(groups):
            text = " ".join(lines).strip()
            if len(text) < _MIN_CLAUSE_CHARS:
                continue
            anchor = text[:60].strip()
            start = doc.full_text.find(anchor, search_start)
            if start == -1:
                start = search_start
            end = start + len(text)
            search_start = max(search_start, start)
            clauses.append(Clause(
                id=f"{doc.doc_id[:6]}-c{i:03d}", text=text, start=start, end=end,
            ))
        return clauses

    def _map_lc_chunks_to_clauses(self, doc: Document, chunks) -> list[Clause]:
        clauses: list[Clause] = []
        search_start = 0
        for i, chunk_doc in enumerate(chunks):
            text = chunk_doc.page_content.strip()
            if len(text) < _MIN_CLAUSE_CHARS:
                continue
            heading_path = [
                chunk_doc.metadata[k]
                for k in ("h1", "h2", "h3")
                if k in chunk_doc.metadata
            ]
            anchor = text[:60].strip()
            start = doc.full_text.find(anchor, search_start)
            if start == -1:
                start = max(search_start, 0)
            end = start + len(text)
            search_start = max(search_start, start)
            clauses.append(Clause(
                id=f"{doc.doc_id[:6]}-c{i:03d}",
                text=text, start=start, end=end, heading_path=heading_path,
            ))
        return clauses

    # ── Anchor-matching normalisation ─────────────────────────────────────────

    def _normalize_with_map(self, text: str) -> tuple[str, list[int]]:
        """
        Return (normalized_text, index_map) where normalized_text is lower-cased,
        quote/dash-canonicalised and whitespace-collapsed, and index_map[k] is the
        offset in the ORIGINAL text of the k-th normalized character.
        """
        norm_chars: list[str] = []
        index_map: list[int] = []
        prev_space = False
        for i, ch in enumerate(text):
            c = _CANON.get(ch, ch)
            if c.isspace():
                if prev_space:
                    continue
                norm_chars.append(" ")
                index_map.append(i)
                prev_space = True
            else:
                norm_chars.append(c.lower())
                index_map.append(i)
                prev_space = False
        index_map.append(len(text))  # sentinel for end-of-text
        return "".join(norm_chars), index_map

    def _normalize_str(self, s: str) -> str:
        return self._normalize_with_map(s)[0].strip()
