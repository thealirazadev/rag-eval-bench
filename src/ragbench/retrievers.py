"""Retrievers: BM25, dense (cosine over embeddings), and hybrid (reciprocal rank fusion)."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx

from ragbench.chunking import Chunk

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


class Retriever(Protocol):
    name: str

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[Chunk]: ...


class BM25:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tfs = [Counter(tokenize(c.text)) for c in self._chunks]
        n = len(self._tfs)
        df: Counter[str] = Counter(term for tf in self._tfs for term in tf)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._avg_len = sum(sum(tf.values()) for tf in self._tfs) / max(n, 1)

    def search(self, query: str, k: int) -> list[Chunk]:
        terms = tokenize(query)
        scores = []
        for i, tf in enumerate(self._tfs):
            length = sum(tf.values())
            score = 0.0
            for t in terms:
                if t in tf:
                    norm = tf[t] + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
                    score += self._idf[t] * tf[t] * (self.k1 + 1) / norm
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._chunks[i] for _, i in scores[:k]]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str], kind: str) -> list[list[float]]: ...


class TrigramEmbedder:
    """Deterministic local embedder: hashed character trigrams of each word.

    It matches morphology ("refund" and "refunds") that BM25 misses, needs no network, and gives
    a reproducible baseline. It is not a semantic model.
    """

    name = "trigram"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str], kind: str) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in tokenize(text):
            padded = f"#{word}#"
            for i in range(len(padded) - 2):
                h = hashlib.blake2b(padded[i : i + 3].encode(), digest_size=4).digest()
                vector[int.from_bytes(h, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class EmbeddingError(Exception):
    """The embedding API could not be reached or refused the request."""


class VoyageEmbedder:
    url = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "voyage-3.5") -> None:
        self.name = f"voyage:{model}"
        self._model = model
        self._client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key}"})

    def embed(self, texts: Sequence[str], kind: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            try:
                response = self._client.post(
                    self.url,
                    json={
                        "input": list(texts[start : start + 128]),
                        "model": self._model,
                        "input_type": "query" if kind == "query" else "document",
                    },
                )
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Voyage API unreachable: {exc}") from exc
            if response.status_code != 200:
                hint = " (check VOYAGE_API_KEY)" if response.status_code in (401, 403) else ""
                raise EmbeddingError(f"Voyage API returned HTTP {response.status_code}{hint}")
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in data)
        return vectors


class Dense:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.name = f"dense:{embedder.name}"
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vectors = self.embedder.embed([c.text for c in self._chunks], "document")

    def search(self, query: str, k: int) -> list[Chunk]:
        q = self.embedder.embed([query], "query")[0]
        scored = sorted(
            (
                (sum(a * b for a, b in zip(q, v, strict=True)), i)
                for i, v in enumerate(self._vectors)
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [self._chunks[i] for _, i in scored[:k]]


class Hybrid:
    """Reciprocal rank fusion of several retrievers: robust without tuning score scales."""

    def __init__(self, parts: Sequence[Retriever], rrf_k: int = 60, depth: int = 50) -> None:
        self.parts = list(parts)
        self.rrf_k, self.depth = rrf_k, depth
        self.name = "hybrid(" + "+".join(p.name for p in self.parts) + ")"

    def index(self, chunks: Sequence[Chunk]) -> None:
        for part in self.parts:
            part.index(chunks)

    def search(self, query: str, k: int) -> list[Chunk]:
        scores: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        # Each part must rank at least k chunks, or a large k could never be filled.
        depth = max(self.depth, k)
        for part in self.parts:
            for rank, chunk in enumerate(part.search(query, depth)):
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[chunk.id] = chunk
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [by_id[cid] for cid in ranked[:k]]
