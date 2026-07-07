"""Segmenter — one LLM call per document to identify side + service line."""
from __future__ import annotations
import json
from models import Document, Segment


class Segmenter:
    def segment(self, doc: Document) -> Segment:
        from llm import get_client, MODEL, llm_available

        if not llm_available():
            return Segment(side="supplier", service_line="general_supplier", confidence=0.1)

        sample = doc.full_text[:2500]
        client = get_client()
        prompt = (
            "You are a contract analyst. Read the following contract excerpt and determine:\n"
            "1. SIDE: Is the subject organisation the BUYER of a service ('supplier') "
            "or the SELLER of a service ('customer')?\n"
            "2. SERVICE_LINE: A short snake_case id for the product/service category, "
            "e.g. 'fuel_cards', 'vehicle_lease', 'freight', 'nda', 'it_services', 'general_supplier'.\n\n"
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
            return Segment(
                side=r.get("side", "supplier"),
                service_line=r.get("service_line", "general_supplier"),
                confidence=float(r.get("confidence", 0.5)),
            )
        except Exception as exc:
            print(f"[segmenter] error: {exc}")
            return Segment(side="supplier", service_line="general_supplier", confidence=0.1)
