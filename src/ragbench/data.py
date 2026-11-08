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


def load_beir(directory: Path, split: str = "test") -> Dataset:
    """Load a BEIR-format dataset: corpus.jsonl, queries.jsonl, and qrels/<split>.tsv.

    Relevance grades above 0 count as relevant (metrics here are binary). Queries without a
    relevant document in the split are dropped, as BEIR's own evaluation does.
    """
    qrels_path = directory / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise DatasetError(f"missing {qrels_path}")
    documents = tuple(
        Document(id=str(row["_id"]), title=str(row.get("title", "")), text=str(row["text"]))
        for row in _read_jsonl(directory / "corpus.jsonl")
    )
    known = {doc.id for doc in documents}
    relevant: dict[str, set[str]] = {}
    for number, line in enumerate(qrels_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if number == 1 and fields[0] == "query-id":
            continue
        if len(fields) != 3:
            raise DatasetError(f"{qrels_path}:{number}: expected query-id, corpus-id, score")
        query_id, doc_id, score = fields
        try:
            graded = int(score) > 0
        except ValueError as exc:
            raise DatasetError(f"{qrels_path}:{number}: score must be an integer") from exc
        if doc_id not in known:
            raise DatasetError(f"{qrels_path}:{number}: unknown document id {doc_id!r}")
        if graded:
            relevant.setdefault(query_id, set()).add(doc_id)
    queries = tuple(
        Query(
            id=str(row["_id"]), text=str(row["text"]), relevant=frozenset(relevant[str(row["_id"])])
        )
        for row in _read_jsonl(directory / "queries.jsonl")
        if str(row["_id"]) in relevant
    )
    if not queries:
        raise DatasetError(f"{qrels_path}: no query has a relevant document")
    return Dataset(name=directory.name, documents=documents, queries=queries)


def load_dataset(directory: Path) -> Dataset:
    """Load ``documents.jsonl`` and ``queries.jsonl`` from a directory and validate the labels.

    A directory holding ``corpus.jsonl`` is read as a BEIR dataset instead (test split).
    """
    if (directory / "corpus.jsonl").exists():
        return load_beir(directory)
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
