# Design decisions

- **Document-level metrics.** Retrievers return chunks, but users need documents. Ranking a
  document by its first chunk keeps chunking strategies comparable: a strategy that produces more
  chunks per document cannot win by repeating the same document in the top k.
- **Fetch 5k chunks, score the top k documents.** After de-duplicating chunks into documents there
  must still be k distinct documents to score.
- **Reciprocal rank fusion for hybrid.** BM25 scores and cosine similarities live on different
  scales. RRF combines ranks, so it needs no score normalization or weight tuning.
- **BM25 implemented here.** About 30 lines with no dependency, and the parameters (k1 = 1.5,
  b = 0.75) are explicit.
- **Deterministic default embedder.** Character trigrams make every run reproducible and free, so
  the test suite and CI need no network. Real embeddings are opt-in through Voyage AI.
- **Strict dataset loading.** A query labeled with an unknown document id is a dataset bug that
  would silently lower recall; the loader fails instead.
- **Bootstrap intervals, not p-values.** A seeded bootstrap over queries needs no distribution
  assumptions or dependencies, reproduces exactly, and answers the practical question: could the
  gap between two configurations be noise? Comparisons are paired (same queries), which is far
  more sensitive than overlapping two unpaired intervals.
- **BEIR as the second dataset format.** It is the common format for public IR benchmarks, so a
  user can test on hundreds of real queries without writing a converter.
