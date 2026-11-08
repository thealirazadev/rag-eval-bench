"""Run every (chunker, retriever) pair over a dataset and average the metrics per query."""

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace

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
from ragbench.stats import confidence_interval, paired_difference_interval


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
    ndcg_ci: tuple[float, float] = (0.0, 0.0)
    # True when the paired difference from the best nDCG is not clearly different from zero.
    ties_best: bool = False
    per_query_ndcg: tuple[float, ...] = field(default=(), repr=False)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("per_query_ndcg")
        data["ndcg_ci"] = list(self.ndcg_ci)
        return data


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


@dataclass(frozen=True)
class Scores:
    chunks: int
    recall: float
    mrr: float
    ndcg: float
    ms_per_query: float
    per_query_ndcg: tuple[float, ...]


def evaluate(dataset: Dataset, chunker: Chunker, retriever: Retriever, k: int) -> Scores:
    chunks = chunker(dataset.documents)
    retriever.index(chunks)
    recall = mrr = 0.0
    ndcgs: list[float] = []
    started = time.perf_counter()
    for query in dataset.queries:
        # Fetch extra chunks so k distinct documents are available after de-duplication.
        ranked = ranked_documents(retriever.search(query.text, k * 5))
        recall += recall_at_k(ranked, query.relevant, k)
        mrr += mrr_at_k(ranked, query.relevant, k)
        ndcgs.append(ndcg_at_k(ranked, query.relevant, k))
    elapsed_ms = (time.perf_counter() - started) * 1000
    n = len(dataset.queries) or 1
    return Scores(len(chunks), recall / n, mrr / n, sum(ndcgs) / n, elapsed_ms / n, tuple(ndcgs))


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
            s = evaluate(dataset, chunker, make_retriever(spec)(), k)
            results.append(
                Result(
                    label,
                    spec,
                    s.chunks,
                    s.recall,
                    s.mrr,
                    s.ndcg,
                    s.ms_per_query,
                    k,
                    confidence_interval(s.per_query_ndcg),
                    per_query_ndcg=s.per_query_ndcg,
                )
            )
    return _mark_ties(results)


def _mark_ties(results: list[Result]) -> list[Result]:
    """Flag configurations whose nDCG is not clearly below the best one on the same queries."""
    if not results:
        return results
    best = max(results, key=lambda r: r.ndcg)
    marked = []
    for r in results:
        low, high = paired_difference_interval(best.per_query_ndcg, r.per_query_ndcg)
        marked.append(replace(r, ties_best=r is best or low <= 0.0 <= high))
    return marked


def to_markdown(results: Sequence[Result]) -> str:
    k = results[0].k if results else 5
    lines = [
        f"| chunking | retriever | chunks | recall@{k} | MRR@{k} | nDCG@{k} | nDCG 95% CI "
        "| vs best | ms/query |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in sorted(results, key=lambda r: (-r.ndcg, r.chunker, r.retriever)):
        low, high = r.ndcg_ci
        lines.append(
            f"| {r.chunker} | {r.retriever} | {r.chunks} | {r.recall:.3f} | {r.mrr:.3f} | "
            f"{r.ndcg:.3f} | {low:.3f}-{high:.3f} | {'tie' if r.ties_best else 'worse'} | "
            f"{r.ms_per_query:.2f} |"
        )
    lines.append(
        "\n'tie': the paired 95% bootstrap interval of the nDCG difference from the best "
        "configuration includes zero, so these queries cannot separate them."
    )
    return "\n".join(lines)
