"""Benchmark datasets: documents plus queries labeled with their relevant document ids."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant: frozenset[str]


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]


class DatasetError(Exception):
    pass


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise DatasetError(f"missing {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise DatasetError(f"{path}:{number}: invalid JSON") from exc
    return rows


def load_dataset(directory: Path) -> Dataset:
    """Load ``documents.jsonl`` and ``queries.jsonl`` from a directory and validate the labels."""
    documents = tuple(
        Document(id=str(row["id"]), title=str(row.get("title", "")), text=str(row["text"]))
        for row in _read_jsonl(directory / "documents.jsonl")
    )
    known = {doc.id for doc in documents}
    if len(known) != len(documents):
        raise DatasetError("document ids must be unique")
    queries = []
    for row in _read_jsonl(directory / "queries.jsonl"):
        relevant = row.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise DatasetError(f"query {row.get('id')}: 'relevant' must be a non-empty list")
        unknown = set(map(str, relevant)) - known
        if unknown:
            raise DatasetError(f"query {row.get('id')}: unknown document ids {sorted(unknown)}")
        queries.append(
            Query(id=str(row["id"]), text=str(row["text"]), relevant=frozenset(map(str, relevant)))
        )
    return Dataset(name=directory.name, documents=documents, queries=tuple(queries))
