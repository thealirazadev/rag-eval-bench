import json
import math
from pathlib import Path

import pytest

from ragbench.bench import make_retriever, run, to_markdown
from ragbench.chunking import Chunk, build_chunker, fixed_words, paragraphs, whole_document
from ragbench.cli import main
from ragbench.data import DatasetError, Document, load_dataset
from ragbench.metrics import mrr_at_k, ndcg_at_k, ranked_documents, recall_at_k
from ragbench.retrievers import BM25, Dense, Hybrid, TrigramEmbedder

ROOT = Path(__file__).parents[1]
DOCS = [
    Document("refunds", "Refunds", "Annual plans are refunded within 14 days."),
    Document("sso", "Single sign-on", "SAML single sign-on works with Okta."),
    Document("export", "Export", "Download every invoice as CSV.\n\nAttachments come as a ZIP."),
]


def chunks(*doc_ids: str) -> list[Chunk]:
    return [Chunk(f"{d}#{i}", d, "") for i, d in enumerate(doc_ids)]


def test_documents_are_ranked_by_their_first_chunk() -> None:
    assert ranked_documents(chunks("a", "a", "b", "a", "c")) == ["a", "b", "c"]


def test_metrics_match_hand_computed_values() -> None:
    ranked, relevant = ["x", "a", "y", "b"], frozenset({"a", "b"})

    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert mrr_at_k(ranked, relevant, 3) == 0.5
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(expected)
    assert mrr_at_k(["x", "y"], relevant, 2) == 0.0


def test_fixed_windows_overlap_and_cover_every_word() -> None:
    doc = Document("d", "", " ".join(f"w{i}" for i in range(10)))

    result = fixed_words(4, overlap=1)([doc])

    assert [c.text.split() for c in result] == [
        ["w0", "w1", "w2", "w3"],
        ["w3", "w4", "w5", "w6"],
        ["w6", "w7", "w8", "w9"],
    ]


def test_paragraphs_merge_short_neighbours_up_to_the_limit() -> None:
    doc = Document("d", "T", "one two\n\nthree four\n\nfive six seven eight nine")

    assert [c.text for c in paragraphs(max_words=5)([doc])] == [
        "T\none two\n\nthree four",
        "T\nfive six seven eight nine",
    ]


def test_bad_chunker_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        fixed_words(5, overlap=5)
    with pytest.raises(ValueError):
        build_chunker({"type": "sentences"})


def test_bm25_prefers_the_document_with_the_query_terms() -> None:
    retriever = BM25()
    retriever.index(whole_document()(DOCS))

    assert [c.doc_id for c in retriever.search("okta single sign-on", 1)] == ["sso"]
    assert retriever.search("nothing matches here", 3) == []


def test_trigram_dense_matches_word_variants_bm25_misses() -> None:
    bm25, dense = BM25(), Dense(TrigramEmbedder())
    for r in (bm25, dense):
        r.index(whole_document()(DOCS))

    assert bm25.search("refund", 1) == []
    assert [c.doc_id for c in dense.search("refund", 1)] == ["refunds"]


def test_hybrid_fuses_rankings_from_both_retrievers() -> None:
    hybrid = Hybrid([BM25(), Dense(TrigramEmbedder())])
    hybrid.index(whole_document()(DOCS))

    assert hybrid.name == "hybrid(bm25+dense:trigram)"
    assert hybrid.search("refund", 1)[0].doc_id == "refunds"


def test_unknown_retriever_specs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="unknown retriever"):
        make_retriever("sparse")()
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        make_retriever("dense:voyage")()


def test_dataset_labels_must_point_at_real_documents(tmp_path: Path) -> None:
    (tmp_path / "documents.jsonl").write_text(json.dumps({"id": "a", "text": "x"}) + "\n")
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "q", "text": "x", "relevant": ["missing"]}) + "\n"
    )

    with pytest.raises(DatasetError, match="unknown document ids"):
        load_dataset(tmp_path)


def test_bundled_benchmark_runs_and_bm25_finds_most_answers() -> None:
    dataset = load_dataset(ROOT / "data" / "help-center")

    results = run(dataset, [{"type": "document"}], ["bm25", "hybrid:trigram"], k=3)

    assert len(dataset.queries) == 20
    assert all(r.recall >= 0.8 for r in results)
    assert to_markdown(results).startswith("| chunking | retriever |")


def test_cli_prints_a_table_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results.json"

    assert main(["run", str(ROOT / "bench.yaml"), "--json", str(out)]) == 0
    assert "recall@3" in capsys.readouterr().out
    assert len(json.loads(out.read_text())) == 9


def test_cli_reports_config_errors(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("dataset: nowhere\n")

    assert main(["run", str(config)]) == 2
