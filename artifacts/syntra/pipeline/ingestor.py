"""Ingestor — bytes → Document (DOCX via python-docx; PDF via LandingAI ADE or pdfplumber)."""
from __future__ import annotations
import io
import os
from models import Document, Element


class Ingestor:
    def ingest(self, file_bytes: bytes, source_type: str, doc_id: str) -> Document:
        if source_type == "docx":
            return self._parse_docx(file_bytes, doc_id)
        elif source_type == "pdf":
            return self._parse_pdf(file_bytes, doc_id)
        raise ValueError(f"Unsupported source type: {source_type!r}")

    # ── DOCX ─────────────────────────────────────────────────────────────────

    def _parse_docx(self, file_bytes: bytes, doc_id: str) -> Document:
        from docx import Document as DocxDoc

        docx = DocxDoc(io.BytesIO(file_bytes))
        elements: list[Element] = []
        text_parts: list[str] = []
        offset = 0
        heading_stack: list[str] = []

        for para in docx.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            kind = "paragraph"
            heading_path = list(heading_stack)

            if para.style.name.startswith("Heading"):
                kind = "heading"
                try:
                    level = int(para.style.name.split()[-1])
                except (IndexError, ValueError):
                    level = 1
                heading_stack = heading_stack[: level - 1] + [text]
                heading_path = list(heading_stack)

            elements.append(Element(
                kind=kind, text=text,
                start=offset, end=offset + len(text),
                heading_path=heading_path,
            ))
            text_parts.append(text)
            offset += len(text) + 1

        full_text = "\n".join(text_parts)
        return Document(doc_id=doc_id, source_type="docx", full_text=full_text, elements=elements)

    # ── PDF ──────────────────────────────────────────────────────────────────

    def _parse_pdf(self, file_bytes: bytes, doc_id: str) -> Document:
        api_key = os.environ.get("LANDING_AI_API_KEY")
        if api_key:
            result = self._parse_pdf_ade(file_bytes, doc_id, api_key)
            if result is not None:
                return result
        return self._parse_pdf_plumber(file_bytes, doc_id)

    def _parse_pdf_ade(self, file_bytes: bytes, doc_id: str, api_key: str) -> Document | None:
        try:
            import tempfile
            from agentic_doc.parse import parse_document  # type: ignore

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                result = parse_document(tmp_path)
                elements: list[Element] = []
                text_parts: list[str] = []
                offset = 0
                for chunk in result.chunks:
                    text = (chunk.text or "").strip()
                    if not text:
                        continue
                    elements.append(Element(
                        kind="paragraph", text=text,
                        start=offset, end=offset + len(text),
                    ))
                    text_parts.append(text)
                    offset += len(text) + 1
                return Document(
                    doc_id=doc_id, source_type="pdf",
                    full_text="\n".join(text_parts), elements=elements,
                )
            finally:
                os.unlink(tmp_path)
        except ImportError:
            return None
        except Exception as exc:
            print(f"[ingestor] ADE error: {exc}")
            return None

    def _parse_pdf_plumber(self, file_bytes: bytes, doc_id: str) -> Document:
        try:
            import pdfplumber  # type: ignore

            elements: list[Element] = []
            text_parts: list[str] = []
            offset = 0
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    raw = page.extract_text() or ""
                    for line in raw.split("\n"):
                        line = line.strip()
                        if len(line) < 3:
                            continue
                        elements.append(Element(
                            kind="paragraph", text=line,
                            start=offset, end=offset + len(line),
                        ))
                        text_parts.append(line)
                        offset += len(line) + 1
        except ImportError:
            placeholder = "[PDF — install pdfplumber or set LANDING_AI_API_KEY]"
            elements = [Element(kind="paragraph", text=placeholder, start=0, end=len(placeholder))]
            text_parts = [placeholder]

        return Document(
            doc_id=doc_id, source_type="pdf",
            full_text="\n".join(text_parts), elements=elements,
        )
