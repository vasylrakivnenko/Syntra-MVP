---
name: legal clause segmentation
description: Why Syntra's chunker segments contracts via LLM-returned anchors, and the rules that keep it correct.
---

# Clause segmentation strategy (Syntra chunker)

`artifacts/syntra/pipeline/chunker.py` splits a parsed contract into top-level `Clause`s. The
primary strategy asks the LLM to return only a short verbatim anchor (first 6-10 words) per
top-level clause; we locate each anchor in the text and split there. Regex/paragraph/character
splitters remain as offline fallbacks used only when no LLM key is present.

**Why LLM anchors (not regex):** pure regex fails on legal contracts because nested numbered
sub-lists restart their own 1,2,3 numbering (e.g. "shall not include: 1. ... 2. ...") and get
mistaken for top-level sections; and PDF parsing sometimes merges several sections into one block or
buries a section start mid-paragraph ("Company agrees as follows: 1. ..."). The LLM understands
structure semantically. Returning only anchors keeps output tokens tiny regardless of document size.

**Rules that keep it correct (learned via review):**
- Match anchors forward-only from the previous hit; never rewind the cursor and never retry-from-start.
  Repeated clause openings otherwise latch onto an earlier duplicate and cascade mis-segmentation.
- Keep matching tolerant: lowercase, collapse whitespace, canonicalize curly quotes/dashes — or valid
  anchors silently fail to locate and clauses get dropped.
- Clause start/end must precisely bracket the *stripped* text within full_text (citations may slice it
  later); account for leading whitespace when computing the real start offset.

**Model:** shared `llm.py` client (OpenAI-compatible Replit AI proxy, `LLM_MODEL` env, default
`gpt-5.1`). Ample context for normal contracts; switch `LLM_MODEL` or wire Anthropic for 1M-token docs.
