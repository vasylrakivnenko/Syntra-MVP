"""Redliner — produce a .docx with colour-coded redline summary section (§9)."""
from __future__ import annotations
import io
from models import Document, ClauseVerdict


_STATUS_COLOR = {
    "unacceptable":        (0xDC, 0x35, 0x45),   # red
    "acceptable_deviation": (0xFF, 0x8C, 0x00),   # orange
    "unusual":             (0x6F, 0x42, 0xC1),    # purple
    "complies":            (0x19, 0x87, 0x54),    # green
}


class Redliner:
    def redline(self, doc: Document, verdicts: list[ClauseVerdict]) -> bytes:
        from docx import Document as DocxDoc
        from docx.shared import RGBColor, Pt

        rdoc = DocxDoc()

        # ── reproduce original document ──────────────────────────────────────
        for el in doc.elements:
            if el.kind == "heading" and el.heading_path:
                level = min(len(el.heading_path), 9)
                rdoc.add_heading(el.text, level=level)
            else:
                rdoc.add_paragraph(el.text)

        # ── redline summary section ──────────────────────────────────────────
        rdoc.add_page_break()
        rdoc.add_heading("REDLINE SUMMARY — ISSUES FLAGGED BY SYNTRA", level=1)

        action_verdicts = [
            v for v in verdicts
            if v.branch == "verdict" and v.status != "complies"
        ]
        silences = [v for v in verdicts if v.branch == "silence"]

        if not action_verdicts and not silences:
            p = rdoc.add_paragraph()
            p.add_run("✓ No material issues found.").bold = True
            return self._to_bytes(rdoc)

        # Missing-clause flags (silence)
        if silences:
            rdoc.add_heading("Missing required clauses", level=2)
            for v in silences:
                p = rdoc.add_paragraph(style="List Bullet")
                run = p.add_run(f"[SILENCE] {v.reason or 'Required clause absent'}")
                run.font.color.rgb = RGBColor(0xDC, 0x35, 0x45)
                run.bold = True

        # Clause-level issues
        if action_verdicts:
            rdoc.add_heading("Clause-level issues", level=2)

        for v in action_verdicts:
            rgb = _STATUS_COLOR.get(v.status or "unusual", (0x6C, 0x75, 0x7D))
            color = RGBColor(*rgb)

            # Status + rule badge
            p = rdoc.add_paragraph()
            tag_run = p.add_run(f"[{(v.status or '').upper().replace('_', ' ')}] ")
            tag_run.bold = True
            tag_run.font.color.rgb = color
            if v.rule_ids:
                rule_run = p.add_run(f"Rule: {', '.join(v.rule_ids)}")
                rule_run.italic = True
                rule_run.font.size = Pt(9)

            # Rationale
            if v.rationale:
                p2 = rdoc.add_paragraph()
                p2.add_run(v.rationale).italic = True

            # Suggested alternative
            if v.suggested_text:
                p3 = rdoc.add_paragraph()
                sugg = p3.add_run(f"SUGGESTION: {v.suggested_text}")
                sugg.font.color.rgb = RGBColor(0x00, 0x56, 0xB3)

            # Separator
            rdoc.add_paragraph()

        return self._to_bytes(rdoc)

    @staticmethod
    def _to_bytes(rdoc) -> bytes:
        buf = io.BytesIO()
        rdoc.save(buf)
        return buf.getvalue()
