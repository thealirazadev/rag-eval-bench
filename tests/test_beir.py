import json
from pathlib import Path

import pytest

from ragbench.data import DatasetError, load_beir, load_dataset


def write_beir(root: Path, qrels: str) -> Path:
    (root / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "Refunds", "text": "Annual plans are refunded within 14 days."},
        {"_id": "d2", "title": "SSO", "text": "SAML single sign-on works with Okta."},
    ]
    queries = [
        {"_id": "q1", "text": "money back on a yearly plan"},
        {"_id": "q2", "text": "okta login"},
        {"_id": "q3", "text": "unjudged query"},
    ]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus))
    (root / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries))
    (root / "qrels" / "test.tsv").write_text(qrels)
    return root


def test_beir_datasets_load_with_graded_qrels(tmp_path: Path) -> None:
    root = write_beir(
        tmp_path / "mini", "query-id\tcorpus-id\tscore\nq1\td1\t2\nq2\td2\t1\nq2\td1\t0\n"
    )

    dataset = load_dataset(root)

    assert dataset.name == "mini"
    assert [d.id for d in dataset.documents] == ["d1", "d2"]
    assert {q.id: q.relevant for q in dataset.queries} == {
        "q1": frozenset({"d1"}),
        "q2": frozenset({"d2"}),
    }


@pytest.mark.parametrize(
    ("qrels", "message"),
    [
        ("q1\td9\t1\n", "unknown document"),
        ("q1\td1\n", "expected query-id"),
        ("q1\td1\thigh\n", "integer"),
        ("q1\td1\t0\n", "no query has a relevant"),
    ],
)
def test_bad_beir_qrels_are_rejected(tmp_path: Path, qrels: str, message: str) -> None:
    root = write_beir(tmp_path / "bad", qrels)
    with pytest.raises(DatasetError, match=message):
        load_beir(root)


def test_a_missing_split_is_named(tmp_path: Path) -> None:
    root = write_beir(tmp_path / "mini", "q1\td1\t1\n")
    with pytest.raises(DatasetError, match=r"dev\.tsv"):
        load_beir(root, split="dev")
