# rag-eval-bench

Measure retrieval before you tune prompts. `ragbench` runs every combination of chunking strategy and
retriever over a labeled dataset and reports recall@k, MRR@k, and nDCG@k, so choices like "paragraph
chunks or fixed windows?" and "is hybrid search worth it here?" get answered with numbers instead
of intuition.

- **Chunkers:** whole document, fixed word windows with overlap, paragraph packing.
- **Retrievers:** BM25 (implemented here, no dependencies), dense (cosine over embeddings), and
  hybrid (reciprocal rank fusion of BM25 and dense).
- **Embedders:** a deterministic character-trigram embedder for offline, reproducible runs, or
  [Voyage AI](https://www.voyageai.com/) (`dense:voyage` / `hybrid:voyage:voyage-3.5`) with
  `VOYAGE_API_KEY` set.
- **Metrics at the document level:** a document's rank is the rank of its first chunk, so splitting
  a document into many chunks cannot crowd the top k and inflate scores.

## Quick start

```sh
uv sync
uv run ragbench run bench.yaml --json results/help-center.json
```

## Results on the bundled dataset

`data/help-center` is a small, fictional software help center: 14 multi-paragraph articles and
20 queries written the way users ask, not the way the articles are worded. With the offline trigram
embedder (k = 3):

| chunking | retriever | chunks | recall@3 | MRR@3 | nDCG@3 | nDCG 95% CI | vs best | ms/query |
|---|---|---:|---:|---:|---:|---|---|---:|
| paragraph | bm25 | 17 | 0.950 | 0.950 | 0.950 | 0.850-1.000 | tie | 0.02 |
| whole-document | bm25 | 14 | 0.950 | 0.950 | 0.950 | 0.850-1.000 | tie | 0.04 |
| paragraph | hybrid:trigram | 17 | 0.950 | 0.950 | 0.946 | 0.842-1.000 | tie | 0.93 |
| fixed-20-overlap-5 | hybrid:trigram | 36 | 0.950 | 0.925 | 0.932 | 0.813-1.000 | tie | 1.87 |
| whole-document | hybrid:trigram | 14 | 0.950 | 0.925 | 0.932 | 0.813-1.000 | tie | 0.79 |
| fixed-20-overlap-5 | bm25 | 36 | 0.950 | 0.900 | 0.913 | 0.795-1.000 | tie | 0.03 |
| fixed-20-overlap-5 | dense:trigram | 36 | 0.925 | 0.925 | 0.912 | 0.793-1.000 | tie | 1.82 |
| paragraph | dense:trigram | 17 | 0.925 | 0.925 | 0.912 | 0.793-1.000 | tie | 0.90 |
| whole-document | dense:trigram | 14 | 0.925 | 0.925 | 0.912 | 0.793-1.000 | tie | 0.79 |

Every interval is a seeded bootstrap over queries (2,000 resamples). "vs best" compares each
configuration with the top one on the same queries: "tie" means the paired 95% interval of the
nDCG difference includes zero.

What this shows, and what it does not:

- **Twenty queries cannot separate these nine configurations.** Every one ties the best. The
  ordering above is a hint, not a result; this is exactly the mistake the intervals are there
  to catch.
- BM25 is about 40 times faster than the dense and hybrid runs here, at no measurable cost in
  quality on this data.
- Hybrid's value is on wording BM25 cannot match: a query for "refund" finds nothing with BM25
  alone but finds the Refunds article with hybrid (see the tests).
- The trigram embedder is a lexical baseline, not a semantic model. Run the same config with
  `dense:voyage` and `hybrid:voyage` to measure what real embeddings add on your data.
- For a decision, use a few hundred labeled queries: your own, or a public BEIR dataset (below).

## Bring your own data

A dataset is a folder with two JSONL files:

```jsonl
// documents.jsonl
{"id": "refunds", "title": "Refunds", "text": "Annual plans can be refunded..."}
// queries.jsonl
{"id": "q2", "text": "can I get my money back after buying a yearly plan", "relevant": ["refunds"]}
```

Every id in `relevant` must exist in `documents.jsonl`; the loader rejects the dataset otherwise.

**BEIR datasets** load as they are: point `dataset` at a folder holding `corpus.jsonl`,
`queries.jsonl`, and `qrels/test.tsv` (for example an unzipped
[BEIR](https://github.com/beir-cellar/beir) dataset such as SciFact or FiQA). Relevance grades
above 0 count as relevant, and queries with no relevant document in the split are skipped.
Then point a config at it:

```yaml
dataset: data/my-docs
k: 5
chunkers:
  - {name: paragraph, type: paragraph, max_words: 120}
  - {name: fixed-200, type: fixed, size: 200, overlap: 40}
retrievers: [bm25, dense:voyage, hybrid:voyage]
```

## Development

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Design notes are in [docs/decisions.md](docs/decisions.md). MIT licensed.
