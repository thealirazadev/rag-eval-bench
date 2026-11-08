"""Chunking strategies. Each turns documents into chunks that remember their source document."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ragbench.data import Document


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


Chunker = Callable[[Iterable[Document]], list[Chunk]]


def whole_document() -> Chunker:
    def chunk(documents: Iterable[Document]) -> list[Chunk]:
        return [Chunk(f"{d.id}#0", d.id, f"{d.title}\n{d.text}".strip()) for d in documents]

    return chunk


def fixed_words(size: int, overlap: int = 0) -> Chunker:
    """Windows of ``size`` words, each starting ``size - overlap`` words after the previous one."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    step = size - overlap

    def chunk(documents: Iterable[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            words = doc.text.split()
            for n, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
                piece = " ".join(words[start : start + size])
                chunks.append(Chunk(f"{doc.id}#{n}", doc.id, f"{doc.title}\n{piece}".strip()))
        return chunks

    return chunk


def paragraphs(max_words: int = 200) -> Chunker:
    """One chunk per paragraph, merging short neighbours up to ``max_words``."""

    def chunk(documents: Iterable[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            current: list[str] = []
            n = 0
            for para in (p.strip() for p in re.split(r"\n\s*\n", doc.text)):
                if not para:
                    continue
                if current and len(" ".join([*current, para]).split()) > max_words:
                    chunks.append(
                        Chunk(f"{doc.id}#{n}", doc.id, f"{doc.title}\n" + "\n\n".join(current))
                    )
                    current, n = [], n + 1
                current.append(para)
            if current:
                chunks.append(
                    Chunk(f"{doc.id}#{n}", doc.id, f"{doc.title}\n" + "\n\n".join(current))
                )
        return chunks

    return chunk


def build_chunker(spec: dict[str, object]) -> Chunker:
    kind = spec.get("type", "document")
    if kind == "document":
        return whole_document()
    if kind == "fixed":
        return fixed_words(int(str(spec.get("size", 100))), int(str(spec.get("overlap", 0))))
    if kind == "paragraph":
        return paragraphs(int(str(spec.get("max_words", 200))))
    raise ValueError(f"unknown chunker type {kind!r}")
