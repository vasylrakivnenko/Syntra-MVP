from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
import uuid


class Element(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    kind: str                    # heading | paragraph | table_cell
    text: str
    start: int
    end: int
    heading_path: list[str] = []


class Document(BaseModel):
    doc_id: str
    source_type: str             # docx | pdf
    full_text: str
    elements: list[Element]


class Clause(BaseModel):
    id: str
    text: str
    start: int
    end: int
    heading_path: list[str] = []


class Classification(BaseModel):
    clause_type: str             # from the playbook taxonomy
    confidence: float
    spans: list[tuple[int, int]] = []


class Segment(BaseModel):        # which matrix ROW an inbound contract belongs to
    side: str                    # supplier (we buy) | customer (we sell)
    service_line: str            # matches a ServiceLine.id
    confidence: float


class Position(BaseModel):       # one matrix CELL — one citable rule
    id: str                      # stable citation ID, e.g. "PD-FUEL-1"
    preferred: str
    fallback: str
    walk_away: str
    risk_weight: Optional[int] = None   # overrides column default when set
    source_doc_ids: list[str] = []


class PolicyColumn(BaseModel):   # a COLUMN of the matrix
    id: str
    label: str
    clause_type: str             # links to runtime taxonomy
    required: bool = False
    risk_weight: int = 3         # column default; cell may override
    cuad_map: Optional[str] = None


class ServiceLine(BaseModel):    # a ROW of the matrix
    id: str
    label: str
    side: str                    # supplier | customer
    positions: dict[str, Position] = {}   # keyed by PolicyColumn.id; missing = gap


class Playbook(BaseModel):
    version: str
    policies: list[PolicyColumn] = []
    service_lines: list[ServiceLine] = []

    def get_policy_by_clause_type(self, clause_type: str) -> Optional[PolicyColumn]:
        return next((p for p in self.policies if p.clause_type == clause_type), None)

    def get_service_line(self, service_line_id: str) -> Optional[ServiceLine]:
        return next((sl for sl in self.service_lines if sl.id == service_line_id), None)

    def lookup(self, service_line_id: str, clause_type: str) -> Optional[Position]:
        """Two-axis deterministic lookup: playbook[service_line][clause_type]."""
        position, _ = self.lookup_resolved(service_line_id, clause_type)
        return position

    def lookup_resolved(
        self, service_line_id: str, clause_type: str
    ) -> tuple[Optional[Position], Optional[str]]:
        """Like lookup(), but also returns the service-line id of the row where the
        position actually lives (differs from the requested row when the lookup falls
        back to general_supplier / general_customer). Citations must point at the
        resolved row, not the requested one.

        If the specific service line has no position for this clause type, fall back
        to the general row for the same side (general_supplier / general_customer) so
        broad clause types (liability, indemnification, IP) stay comparable even for
        specialised rows like a standalone NDA.
        """
        policy = self.get_policy_by_clause_type(clause_type)
        if policy is None:
            return None, None
        sl = self.get_service_line(service_line_id)
        if sl is None:
            return None, None
        position = sl.positions.get(policy.id)
        if position is not None:
            return position, sl.id
        general_id = "general_customer" if sl.side == "customer" else "general_supplier"
        if general_id != service_line_id:
            general = self.get_service_line(general_id)
            if general is not None:
                position = general.positions.get(policy.id)
                if position is not None:
                    return position, general.id
        return None, None


class ClauseVerdict(BaseModel):
    clause_id: str
    branch: str                  # verdict | silence | abstain
    status: Optional[str] = None # complies | acceptable_deviation | unacceptable | unusual
    rule_ids: list[str] = []
    spans: list[tuple[int, int]] = []
    rationale: str = ""
    reason: Optional[str] = None # for abstain / silence
    service_line: Optional[str] = None
    suggested_text: str = ""
    # Snapshot of the playbook position (or required policy, for silence) this
    # verdict was judged against, captured at analysis time so citations stay
    # accurate even after the playbook is edited. See Triage for the shape.
    cited_position: Optional[dict] = None


class QueueItem(BaseModel):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    doc_id: str
    priority: int = 0
    assignee: Optional[str] = None
    status: str = "pending"      # pending | approved | rejected
    reason: Optional[str] = None  # human-readable escalation reason


class Citation(BaseModel):
    rule_id: Optional[str] = None
    doc_id: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    ts: str
    actor_id: str
    action: str
    input_hash: str = ""
    prompt_hash: str = ""
    output_json: str = ""
    citations: list[Citation] = []
    prev_hash: str = ""


class RiskSummaryItem(BaseModel):
    clause_type: str
    status: str
    summary: str
    rule_ids: list[str] = []
    service_line: Optional[str] = None
