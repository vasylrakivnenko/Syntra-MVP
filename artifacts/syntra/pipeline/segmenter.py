"""Segmenter — one LLM call per document to identify side + service line."""
from __future__ import annotations
import json
from typing import Optional
from models import Document, Segment, Playbook


class Segmenter:
    def segment(self, doc: Document, playbook: Optional[Playbook] = None) -> Segment:
        from llm import get_client, MODEL, llm_available

        if not llm_available():
            return Segment(side="supplier", service_line="general_supplier", confidence=0.1)

        sample = doc.full_text[:2500]
        client = get_client()

        # Enumerate the playbook's actual rows so the model MUST pick an existing
        # service line id. A free-form id (e.g. "nda") matches no row, and every
        # clause then abstains at lookup time.
        valid_ids: list[str] = []
        lines_desc = ""
        if playbook and playbook.service_lines:
            valid_ids = [sl.id for sl in playbook.service_lines]
            lines_desc = "\n".join(
                f"  - {sl.id} ({sl.side}): {sl.label}" for sl in playbook.service_lines
            )

        if lines_desc:
            service_line_instr = (
                "2. SERVICE_LINE: Choose the SINGLE best-matching service line id from the "
                "playbook rows below. Prefer the most specific row that fits; use a "
                "'general_*' row only when nothing more specific applies.\n"
                f"{lines_desc}\n"
            )
        else:
            service_line_instr = (
                "2. SERVICE_LINE: A short snake_case id for the product/service category, "
                "e.g. 'nda', 'it_services', 'general_supplier'.\n"
            )

        prompt = (
            "You are a contract analyst. Read the following contract excerpt and determine:\n"
            "1. SIDE: Is the subject organisation the BUYER of a service ('supplier') "
            "or the SELLER of a service ('customer')?\n"
            f"{service_line_instr}\n"
            f"CONTRACT:\n{sample}\n\n"
            "Return JSON:\n"
            '{"side":"supplier|customer","service_line":"<id>","confidence":0.0,"reasoning":"<one sentence>"}'
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            r = json.loads(resp.choices[0].message.content)
            side = r.get("side", "supplier")
            service_line = r.get("service_line", "general_supplier")
            confidence = float(r.get("confidence", 0.5))

            # Guardrail: if the model returned an id that isn't a real row, fall back
            # to a genuine row for the detected side rather than causing abstains.
            # Never assign an id that isn't in the playbook (that is the original bug).
            if valid_ids and service_line not in valid_ids:
                preferred = "general_customer" if side == "customer" else "general_supplier"
                if preferred in valid_ids:
                    service_line = preferred
                else:
                    same_side = [sl.id for sl in playbook.service_lines if sl.side == side]
                    service_line = same_side[0] if same_side else valid_ids[0]

            # Keep side consistent with the resolved row.
            if playbook:
                sl = playbook.get_service_line(service_line)
                if sl is not None:
                    side = sl.side

            return Segment(side=side, service_line=service_line, confidence=confidence)
        except Exception as exc:
            print(f"[segmenter] error: {exc}")
            return Segment(side="supplier", service_line="general_supplier", confidence=0.1)
