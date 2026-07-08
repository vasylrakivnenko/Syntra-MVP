# Data attribution

The prebuilt market table (`contractnli_table/`) and extracted records
(`records_cnli/`) are derived from the **ContractNLI** dataset:

> Yuta Koreeda and Christopher D. Manning. *ContractNLI: A Dataset for
> Document-level Natural Language Inference for Contracts.* Findings of EMNLP 2021.
> Hitachi America, Ltd. — https://stanfordnlp.github.io/contract-nli/

**License: Creative Commons Attribution 4.0 International (CC BY 4.0).**
Redistribution and commercial use are permitted with attribution.

Only the **SEC-sourced subset** (`document_type` = `sec-text` / `sec-html`, i.e.
already-public SEC filings) was used for the shipped table — 200 of the 232
SEC-sourced NDAs across the train/dev/test splits. The records contain
LegaLens-derived typed field values plus short verbatim evidence spans.

Note for counsel: CC BY 4.0 covers the dataset/compilation; the underlying
documents' copyright is separate. The SEC-sourced subset used here consists of
already-public filings. The web-sourced ContractNLI subset (`search-pdf`) was
**not** used.
