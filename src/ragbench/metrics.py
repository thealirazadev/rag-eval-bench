"""Ranking metrics over documents (binary relevance).

Retrievers return chunks; a document's rank is the rank of its first chunk, so a document split
into many chunks cannot crowd the top k and inflate scores.
"""

import math
from collections.abc import Sequence

from ragbench.chunking import Chunk


def ranked_documents(chunks: Sequence[Chunk]) -> list[str]:
    seen: list[str] = []
    for chunk in chunks:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
    return seen


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
