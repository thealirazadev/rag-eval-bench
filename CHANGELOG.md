# Changelog

## 0.2.0 - 2026-11-09

- Bootstrap 95% confidence intervals for nDCG, and a paired comparison that marks each
  configuration as a tie with the best or worse. On the bundled 20 queries every configuration
  ties, so the README no longer draws conclusions the data cannot support.
- BEIR-format datasets (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) load directly.
- Embedding API failures (bad key, outage) print a clear error instead of a traceback.
- CI on Linux, macOS, and Windows with Python 3.11 and 3.13; coverage gate at 95 percent.

## 0.1.0

Chunkers (document, fixed windows, paragraphs), BM25, dense, and hybrid retrievers, recall/MRR/
nDCG at the document level, YAML configs, JSON output.
