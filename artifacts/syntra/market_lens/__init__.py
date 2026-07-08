"""Market Lens — statistical layer that types each NDA into a market table and
scores new NDAs on per-field percentiles + an Off-Market Index.

Pipeline: extract (LLM) -> build_table -> stats. No UI: `python -m market_lens.score <file>`.
"""

__version__ = "0.1.0"
