import pytest

from ragbench.bench import Result, _mark_ties, run, to_markdown
from ragbench.data import Dataset, Document, Query
from ragbench.stats import confidence_interval, paired_difference_interval


def test_interval_brackets_the_mean_and_is_reproducible() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0]
    low, high = confidence_interval(scores)

    assert low <= sum(scores) / len(scores) <= high
    assert confidence_interval(scores) == (low, high)
    assert confidence_interval([0.7] * 10) == pytest.approx((0.7, 0.7))
    assert confidence_interval([]) == (0.0, 0.0)


def test_paired_difference_separates_a_clear_gap_and_not_noise() -> None:
    better, worse = [1.0] * 30, [0.0] * 30
    low, _ = paired_difference_interval(better, worse)
    assert low > 0

    noisy_a = [1.0, 0.0] * 15
    noisy_b = [0.0, 1.0] * 15
    low, high = paired_difference_interval(noisy_a, noisy_b)
    assert low <= 0 <= high

    with pytest.raises(ValueError):
        paired_difference_interval([1.0], [1.0, 0.0])
    assert paired_difference_interval([], []) == (0.0, 0.0)


def result(name: str, scores: list[float]) -> Result:
    mean = sum(scores) / len(scores)
    return Result("doc", name, 1, mean, mean, mean, 0.1, 1, per_query_ndcg=tuple(scores))


def test_the_report_marks_a_clearly_worse_configuration() -> None:
    best = result("best", [1.0] * 30)
    close = result("close", [1.0] * 29 + [0.0])
    worse = result("worse", [1.0, 0.0] * 15)
    marked = {r.retriever: r for r in _mark_ties([close, best, worse])}

    assert marked["best"].ties_best and marked["close"].ties_best
    assert not marked["worse"].ties_best
    assert "per_query_ndcg" not in marked["best"].as_dict()
    assert "| worse |" in to_markdown(list(marked.values()))
    assert _mark_ties([]) == []


def test_run_reports_intervals_for_every_configuration() -> None:
    docs = tuple(Document(f"d{i}", "", f"code{i}x{i}z alpha") for i in range(10))
    queries = tuple(Query(f"q{i}", f"code{i}x{i}z", frozenset({f"d{i}"})) for i in range(10))
    results = run(
        Dataset("synthetic", docs, queries),
        [{"name": "doc", "type": "document"}],
        ["bm25", "dense:trigram"],
        k=1,
    )

    assert all(r.as_dict()["ndcg_ci"] == [1.0, 1.0] for r in results)


def test_a_lower_level_gives_a_narrower_interval_inside_the_wider_one() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.25, 0.75]
    wide_low, wide_high = confidence_interval(scores, level=0.95)
    narrow_low, narrow_high = confidence_interval(scores, level=0.5)

    assert wide_low <= narrow_low <= narrow_high <= wide_high
    assert narrow_high - narrow_low < wide_high - wide_low
