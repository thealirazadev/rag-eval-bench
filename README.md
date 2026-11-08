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

| chunking | retriever | chunks | recall@3 | MRR@3 | nDCG@3 | ms/query |
|---|---|---:|---:|---:|---:|---:|
| paragraph | bm25 | 17 | 0.950 | 0.950 | 0.950 | 0.01 |
| whole-document | bm25 | 14 | 0.950 | 0.950 | 0.950 | 0.01 |
| paragraph | hybrid:trigram | 17 | 0.950 | 0.950 | 0.946 | 0.48 |
| fixed-20-overlap-5 | hybrid:trigram | 36 | 0.950 | 0.925 | 0.932 | 0.95 |
| whole-document | hybrid:trigram | 14 | 0.950 | 0.925 | 0.932 | 0.39 |
| fixed-20-overlap-5 | bm25 | 36 | 0.950 | 0.900 | 0.913 | 0.02 |
| fixed-20-overlap-5 | dense:trigram | 36 | 0.925 | 0.925 | 0.912 | 0.92 |
| paragraph | dense:trigram | 17 | 0.925 | 0.925 | 0.912 | 0.46 |
| whole-document | dense:trigram | 14 | 0.925 | 0.925 | 0.912 | 0.39 |

What this shows, and what it does not:

- On short, keyword-rich articles BM25 is hard to beat. It is also about 40 times faster here.
- Hybrid only ties BM25 on this dataset. Its value is on wording BM25 cannot match: a query for
  "refund" finds nothing with BM25 alone but finds the Refunds article with hybrid (see the tests).
- Small fixed windows cut answers in half and lower MRR. Chunk size is a retrieval decision, not
  just a context-window one.
- The trigram embedder is a lexical baseline, not a semantic model. Run the same config with
  `dense:voyage` and `hybrid:voyage` to measure what real embeddings add on your data.
- 20 queries is a smoke test. Use the format below to benchmark your own corpus before deciding.

## Bring your own data

A dataset is a folder with two JSONL files:

```jsonl
// documents.jsonl
{"id": "refunds", "title": "Refunds", "text": "Annual plans can be refunded..."}
// queries.jsonl
{"id": "q2", "text": "can I get my money back after buying a yearly plan", "relevant": ["refunds"]}
```

Every id in `relevant` must exist in `documents.jsonl`; the loader rejects the dataset otherwise.
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
