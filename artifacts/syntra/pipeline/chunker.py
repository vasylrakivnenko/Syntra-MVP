"""Chunker — Document → list[Clause] using langchain-text-splitters (§4).

Strategy:
  1. Convert Document elements to Markdown with heading markers.
  2. MarkdownHeaderTextSplitter splits by heading hierarchy (ADE / python-docx structure).
  3. Each chunk maps back to Clause with exact character offsets.
  Fallback: RecursiveCharacterTextSplitter when no headings are detected.
"""
from __future__ import annotations
from models import Document, Clause


class Chunker:
    def chunk(self, doc: Document) -> list[Clause]:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        # Build Markdown from document elements
        md_lines: list[str] = []
        for el in doc.elements:
            if el.kind == "heading" and el.heading_path:
                level = min(len(el.heading_path), 3)
                md_lines.append(f"{'#' * level} {el.text}")
            else:
                md_lines.append(el.text)
        md_text = "\n\n".join(md_lines)

        headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split, strip_headers=False
        )
        chunks = header_splitter.split_text(md_text)

        # Filter empty chunks and delegate to fallback when no headings found
        non_empty = [c for c in chunks if len(c.page_content.strip()) >= 30]
        if len(non_empty) <= 1:
            return self._fallback_split(doc)

        return self._map_to_clauses(doc, non_empty)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _map_to_clauses(self, doc: Document, chunks) -> list[Clause]:
        """Map LangChain chunk documents to Clause objects with best-effort offsets."""
        clauses: list[Clause] = []
        search_start = 0
        for i, chunk_doc in enumerate(chunks):
            text = chunk_doc.page_content.strip()
            if len(text) < 30:
                continue
            heading_path = [
                chunk_doc.metadata[k]
                for k in ("h1", "h2", "h3")
                if k in chunk_doc.metadata
            ]
            # Find offset in full_text using a short anchor
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

    def _fallback_split(self, doc: Document) -> list[Clause]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        chunks = splitter.split_text(doc.full_text)
        clauses: list[Clause] = []
        search_start = 0
        for i, text in enumerate(chunks):
            if len(text.strip()) < 30:
                continue
            anchor = text[:50].strip()
            start = doc.full_text.find(anchor, search_start)
            if start == -1:
                start = search_start
            end = start + len(text)
            search_start = max(search_start, start)
            clauses.append(Clause(
                id=f"{doc.doc_id[:6]}-c{i:03d}",
                text=text.strip(),
                start=start,
                end=end,
            ))
        return clauses
