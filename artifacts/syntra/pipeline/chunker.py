"""Chunker — Document → list[Clause].

Splitting priority:
  1. MarkdownHeaderTextSplitter  — for DOCX files with heading styles
  2. Numbered-section splitter   — recognises "1. Title" / "(a)" clause starts
  3. Paragraph-boundary splitter — groups elements into clauses by blank-line gaps
  4. RecursiveCharacterTextSplitter — last resort for dense unstructured text
"""
from __future__ import annotations
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

_MIN_CLAUSE_CHARS = 60   # ignore chunks shorter than this


class Chunker:
    def chunk(self, doc: Document) -> list[Clause]:
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
        Group document elements by numbered section headings.
        Any element matching _SECTION_START begins a new clause group.
        """
        groups: list[list[str]] = []
        current: list[str] = []

        for el in doc.elements:
            if _SECTION_START.match(el.text) and current:
                groups.append(current)
                current = [el.text]
            else:
                current.append(el.text)
        if current:
            groups.append(current)

        # If no split happened (only 1 group = entire doc), bail out
        if len(groups) <= 1:
            return []

        return self._groups_to_clauses(doc, groups)

    # ── Strategy 3: paragraph boundaries ─────────────────────────────────────

    def _split_by_paragraphs(self, doc: Document) -> list[Clause]:
        """
        Treat each Document element as a paragraph-level clause.
        Merge very short ones (< 120 chars) with the previous clause.
        """
        merged: list[list[str]] = []
        for el in doc.elements:
            text = el.text.strip()
            if not text:
                continue
            if merged and len(text) < 120:
                merged[-1].append(text)
            else:
                merged.append([text])

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
        for i, text in enumerate(chunks):
            text = text.strip()
            if len(text) < _MIN_CLAUSE_CHARS:
                continue
            anchor = text[:50].strip()
            start = doc.full_text.find(anchor, search_start)
            if start == -1:
                start = search_start
            end = start + len(text)
            search_start = max(search_start, start)
            clauses.append(Clause(
                id=f"{doc.doc_id[:6]}-c{i:03d}",
                text=text,
                start=start,
                end=end,
            ))
        return clauses

    # ── Helpers ───────────────────────────────────────────────────────────────

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
                id=f"{doc.doc_id[:6]}-c{i:03d}",
                text=text,
                start=start,
                end=end,
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
                text=text,
                start=start,
                end=end,
                heading_path=heading_path,
            ))
        return clauses
