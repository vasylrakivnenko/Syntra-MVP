"""Optional TabPFN signal for Market Lens v2 — kept in its own module so
pipeline/market.py never imports pandas/tabpfn_client unless the signal is
actually enabled.

Enabled only when TABPFN_TOKEN is set in the environment. Everything here is
failure-soft by contract: any problem (no token, network down, fit error)
returns None and the evidence bundle simply carries tabpfn_p_obs=None — the
rule-based signal has no external dependency and must never be dragged down
by this one (spec.md §7).

Cost model (spec.md §7, do not violate):
  - fit_all_fields is EXPENSIVE (18 network calls, uploads the whole
    population). It runs at most ONCE per process, lazily on the first NDA
    scored, guarded by a lock. A failed fit is cached so a flaky network
    doesn't re-trigger the expensive path on every upload.
  - score_new_doc is cheap and runs once per incoming NDA.
"""
from __future__ import annotations

import os
import threading
import traceback
from typing import Any

_lock = threading.Lock()
_fitted: dict[str, Any] | None = None
_fit_failed = False


def tabpfn_enabled() -> bool:
    return bool(os.environ.get("TABPFN_TOKEN"))


def tabpfn_signal(schema, target_row: dict[str, Any], table_dir) -> dict[str, dict[str, Any]] | None:
    """Per-field conditional probabilities for one NDA, or None if the signal
    is disabled or unavailable. Never raises."""
    global _fitted, _fit_failed
    if not tabpfn_enabled():
        return None
    try:
        from market_lens.tabpfn_score import fit_all_fields, score_new_doc
    except Exception:
        return None
    with _lock:
        if _fitted is None and not _fit_failed:
            try:
                _fitted = fit_all_fields(table_dir, schema)
            except Exception:
                _fit_failed = True  # don't re-pay the expensive path per doc
                print(f"[market-lens] TabPFN fit failed — continuing rule-only:\n"
                      f"{traceback.format_exc()}")
        fitted = _fitted
    if not fitted:
        return None
    try:
        return score_new_doc(schema, fitted, target_row)
    except Exception:
        print(f"[market-lens] TabPFN scoring failed for one doc — rule-only:\n"
              f"{traceback.format_exc()}")
        return None
