"""Statistics layer (D6).

Two frames:
  - Univariate  = the parity layer: per-field frequency (bool/enum) or
                  percentile (numeric) within the segment.
  - Multivariate = the product: the Off-Market Index v1, built from 2- and
                  3-way clause-COMBINATION rarity within the segment
                  (association-rules style), fully explainable by construction.

Design decisions baked in (see project memory):
  * Segment on MUTUALITY ONLY. Industry is a field, not a scoring segment
    (over-segmentation manufactures false outliers).
  * The rules layer is NOT statistically bulletproof at small N — a combo can
    be rare just because the segment is small. It fails TRANSPARENTLY: we apply
    a min-segment-N floor and a min-marginal-support floor, and label anything
    below threshold "insufficient data" rather than "off-market".
  * The global ML scorer (isolation forest) is deferred to v2.1 and must beat
    this baseline on Precision@20 to earn its opacity. It is NOT implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .schema_loader import Schema, token

# Floors (transparent failure — see module docstring).
MIN_SEGMENT_N = 30          # below this, don't score the combination layer at all
MIN_MARGINAL_SUPPORT = 0.10  # each field-value in a combo must be at least this common
OFF_MARKET_THRESHOLD = 0.05  # combos at/under this segment frequency are flagged off-market
TOP_K_CONTRIBUTIONS = 5


def segment(rows: list[dict[str, Any]], mutual_value: Any) -> list[dict[str, Any]]:
    """Segment on mutuality only. mutual_value None -> whole population."""
    if mutual_value is None:
        return rows
    return [r for r in rows if r.get("mutual") == mutual_value]


# --------------------------------------------------------------------------- #
# Univariate (parity)
# --------------------------------------------------------------------------- #

def univariate(schema: Schema, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-field summary within `rows`. Null values are excluded from the
    denominator and reported as coverage."""
    out: dict[str, dict[str, Any]] = {}
    n = len(rows)
    for f in schema.fields:
        vals = [r.get(f.id) for r in rows]
        present = [v for v in vals if v is not None]
        cov = len(present) / n if n else 0.0
        entry: dict[str, Any] = {"coverage": round(cov, 3), "n_present": len(present)}
        if f.type in ("bool", "enum"):
            freq: dict[str, float] = {}
            for v in present:
                key = "true" if v is True else "false" if v is False else str(v)
                freq[key] = freq.get(key, 0) + 1
            entry["freq"] = {k: round(c / len(present), 3) for k, c in freq.items()} if present else {}
        else:  # numeric_months
            entry["values"] = sorted(float(v) for v in present)
        out[f.id] = entry
    return out


def percentile(sorted_vals: list[float], value: float) -> float:
    """Fraction of sorted_vals <= value (0..1). Empty -> 0."""
    if not sorted_vals:
        return 0.0
    below = sum(1 for v in sorted_vals if v <= value)
    return below / len(sorted_vals)


# --------------------------------------------------------------------------- #
# Multivariate (Off-Market Index v1)
# --------------------------------------------------------------------------- #

def _row_tokens(schema: Schema, row: dict[str, Any]) -> set[str]:
    toks = set()
    for f in schema.fields:
        t = token(f, row.get(f.id))
        if t is not None:
            toks.add(t)
    return toks


@dataclass
class Contribution:
    combo: tuple[str, ...]
    support: float          # fraction of segment sharing this whole combo
    count: int              # docs in segment sharing it
    off_market: bool        # support <= OFF_MARKET_THRESHOLD


@dataclass
class OffMarketResult:
    index: float | None                 # 0..100, or None if insufficient data
    segment_n: int
    status: str                         # "scored" | "insufficient_data"
    contributions: list[Contribution]   # rarest combos present in the target


# --------------------------------------------------------------------------- #
# Off-Market Index v1.1 — magnitude-weighted surprise, rank-normalized
# --------------------------------------------------------------------------- #
# WHY v1.1: the v1 raw-rarity index (off_market_index below) SATURATES — on 100
# real NDAs, 83/100 scored exactly 100/100, because a unique 3-way combo of ~25
# fields is trivially common at N~50. v1.1 instead scores by how many expected
# co-occurrences are MISSING: for a combo whose members are each common enough
# that we'd expect it >= EXP_MIN times under independence, surprise =
# expected - observed. Taking the max surprise per doc and rank-normalizing
# across the segment gives a spread 0-100 distribution.
#
# CAVEAT: MIN_MARGINAL_V11 / EXP_MIN were tuned once on the 100-doc ContractNLI
# sample — NOT validated. The spec's D7 (attorney Precision@20) is the gate
# before trusting this ranking.

MIN_MARGINAL_V11 = 0.15
EXP_MIN = 2.0


@dataclass
class OMScore:
    doc: Any
    index: float                     # 0..100, rank-normalized within segment
    raw: float                       # raw max-surprise (expected - observed)
    contributions: list[tuple]       # (combo, observed, expected), most-surprising first


def _doc_surprise(schema: Schema, target: dict[str, Any],
                  seg_token_sets: list[set[str]], marg: dict[str, float],
                  *, min_marginal: float, exp_min: float, top_k: int):
    n = len(seg_token_sets)
    cand = [t for t in _row_tokens(schema, target) if marg.get(t, 0.0) >= min_marginal]
    scored: list[tuple] = []
    for size in (2, 3):
        for combo in combinations(sorted(cand), size):
            expected = n
            for t in combo:
                expected *= marg[t]
            if expected < exp_min:  # must be common enough that its absence is meaningful
                continue
            observed = sum(1 for s in seg_token_sets if set(combo) <= s)
            surprise = expected - observed
            if surprise > 0:
                scored.append((surprise, combo, observed, expected))
    scored.sort(key=lambda x: -x[0])
    raw = scored[0][0] if scored else 0.0
    contribs = [(c, o, e) for _, c, o, e in scored[:top_k]]
    return raw, contribs


def off_market_rank(
    schema: Schema,
    seg_rows: list[dict[str, Any]],
    *,
    min_segment_n: int = MIN_SEGMENT_N,
    min_marginal: float = MIN_MARGINAL_V11,
    exp_min: float = EXP_MIN,
    top_k: int = TOP_K_CONTRIBUTIONS,
) -> list[OMScore] | None:
    """Score every doc in one (already-segmented) population and rank-normalize.
    Each doc is scored against the OTHER docs in the segment (self excluded).
    Returns None if the segment is below the N floor."""
    n = len(seg_rows)
    if n < min_segment_n:
        return None
    token_sets = [_row_tokens(schema, r) for r in seg_rows]

    raws: list[tuple[float, dict, list]] = []
    for i, target in enumerate(seg_rows):
        others = token_sets[:i] + token_sets[i + 1:]
        m = len(others)
        marg: dict[str, float] = {}
        for s in others:
            for t in s:
                marg[t] = marg.get(t, 0) + 1
        marg = {t: c / m for t, c in marg.items()}
        raw, contribs = _doc_surprise(schema, target, others, marg,
                                      min_marginal=min_marginal, exp_min=exp_min, top_k=top_k)
        raws.append((raw, target, contribs))

    sorted_raw = sorted(r for r, _, _ in raws)
    import bisect
    out = []
    for raw, target, contribs in raws:
        idx = round(100.0 * bisect.bisect_right(sorted_raw, raw) / len(sorted_raw), 1)
        out.append(OMScore(doc=target, index=idx, raw=round(raw, 2), contributions=contribs))
    out.sort(key=lambda x: -x.index)
    return out


def off_market_index(
    schema: Schema,
    target: dict[str, Any],
    seg_rows: list[dict[str, Any]],
    *,
    min_segment_n: int = MIN_SEGMENT_N,
    min_marginal: float = MIN_MARGINAL_SUPPORT,
    threshold: float = OFF_MARKET_THRESHOLD,
    top_k: int = TOP_K_CONTRIBUTIONS,
) -> OffMarketResult:
    """Rank the target's 2- and 3-way clause combinations by how rare each is in
    the segment. The rarest surviving combo drives the index; every scored combo
    is emitted as an explainable contribution string upstream.

    A combo is only considered if EACH of its constituent tokens clears the
    marginal-support floor — so a flagged combo is genuinely a rare *combination*
    of common terms, not an artifact of one rare field.
    """
    n = len(seg_rows)
    if n < min_segment_n:
        return OffMarketResult(index=None, segment_n=n, status="insufficient_data",
                               contributions=[])

    seg_token_sets = [_row_tokens(schema, r) for r in seg_rows]

    def marginal(tok: str) -> float:
        return sum(1 for s in seg_token_sets if tok in s) / n

    target_tokens = sorted(_row_tokens(schema, target))
    # keep only tokens common enough on their own to anchor a combo
    anchorable = [t for t in target_tokens if marginal(t) >= min_marginal]

    contributions: list[Contribution] = []
    for size in (2, 3):
        for combo in combinations(anchorable, size):
            cset = set(combo)
            count = sum(1 for s in seg_token_sets if cset <= s)
            support = count / n
            contributions.append(
                Contribution(combo=combo, support=support, count=count,
                             off_market=support <= threshold)
            )

    if not contributions:
        return OffMarketResult(index=None, segment_n=n, status="insufficient_data",
                               contributions=[])

    contributions.sort(key=lambda c: c.support)
    rarest = contributions[0].support
    # Raw rarity index. NOTE: not yet rank-normalized across the corpus — true
    # 0..100 rank-normalization requires scoring every corpus doc (v0.3 work).
    index = round(100.0 * (1.0 - rarest), 1)
    return OffMarketResult(index=index, segment_n=n, status="scored",
                           contributions=contributions[:top_k])
