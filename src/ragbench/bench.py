"""Run every (chunker, retriever) pair over a dataset and average the metrics per query."""

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from ragbench.chunking import Chunker, build_chunker
from ragbench.data import Dataset
from ragbench.metrics import mrr_at_k, ndcg_at_k, ranked_documents, recall_at_k
from ragbench.retrievers import (
    BM25,
    Dense,
    Embedder,
    Hybrid,
    Retriever,
    TrigramEmbedder,
    VoyageEmbedder,
)


@dataclass(frozen=True)
class Result:
    chunker: str
    retriever: str
    chunks: int
    recall: float
    mrr: float
    ndcg: float
    ms_per_query: float
    k: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def make_embedder(name: str) -> Embedder:
    if name == "trigram":
        return TrigramEmbedder()
    if name.startswith("voyage"):
        key = os.environ.get("VOYAGE_API_KEY", "")
        if not key:
            raise ValueError("VOYAGE_API_KEY is required for voyage embedders")
        model = name.split(":", 1)[1] if ":" in name else "voyage-3.5"
        return VoyageEmbedder(key, model)
    raise ValueError(f"unknown embedder {name!r}")


def make_retriever(spec: str) -> Callable[[], Retriever]:
    """Specs: ``bm25``, ``dense:<embedder>``, or ``hybrid:<embedder>`` (BM25 plus dense)."""

    def build() -> Retriever:
        if spec == "bm25":
            return BM25()
        kind, _, embedder = spec.partition(":")
        if kind == "dense":
            return Dense(make_embedder(embedder or "trigram"))
        if kind == "hybrid":
            return Hybrid([BM25(), Dense(make_embedder(embedder or "trigram"))])
        raise ValueError(f"unknown retriever {spec!r}")

    return build


def evaluate(dataset: Dataset, chunker: Chunker, retriever: Retriever, k: int) -> tuple[float, ...]:
    chunks = chunker(dataset.documents)
    retriever.index(chunks)
    recall = mrr = ndcg = 0.0
    started = time.perf_counter()
    for query in dataset.queries:
        # Fetch extra chunks so k distinct documents are available after de-duplication.
        ranked = ranked_documents(retriever.search(query.text, k * 5))
        recall += recall_at_k(ranked, query.relevant, k)
        mrr += mrr_at_k(ranked, query.relevant, k)
        ndcg += ndcg_at_k(ranked, query.relevant, k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    n = len(dataset.queries) or 1
    return len(chunks), recall / n, mrr / n, ndcg / n, elapsed_ms / n


def run(
    dataset: Dataset,
    chunkers: Sequence[dict[str, object]],
    retrievers: Sequence[str],
    k: int = 5,
) -> list[Result]:
    results = []
    for chunk_spec in chunkers:
        chunker = build_chunker(chunk_spec)
        label = str(chunk_spec.get("name") or chunk_spec.get("type", "document"))
        for spec in retrievers:
            n, recall, mrr, ndcg, ms = evaluate(dataset, chunker, make_retriever(spec)(), k)
            results.append(Result(label, spec, int(n), recall, mrr, ndcg, ms, k))
    return results


def to_markdown(results: Sequence[Result]) -> str:
    k = results[0].k if results else 5
    lines = [
        f"| chunking | retriever | chunks | recall@{k} | MRR@{k} | nDCG@{k} | ms/query |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: (-r.ndcg, r.chunker, r.retriever)):
        lines.append(
            f"| {r.chunker} | {r.retriever} | {r.chunks} | {r.recall:.3f} | {r.mrr:.3f} | "
            f"{r.ndcg:.3f} | {r.ms_per_query:.2f} |"
        )
    return "\n".join(lines)
