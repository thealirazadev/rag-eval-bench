"""Bootstrap confidence intervals over per-query scores.

Small query sets make close configurations indistinguishable; resampling queries shows how much
of a gap is noise. Seeded, so a run reproduces exactly.
"""

import random
from collections.abc import Sequence

RESAMPLES = 2000
SEED = 13


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resampled_means(columns: Sequence[Sequence[float]], resamples: int, seed: int) -> list[float]:
    """Means of (first column minus second, or the single column) over resampled query indices."""
    rng = random.Random(seed)
    n = len(columns[0])
    means = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        if len(columns) == 1:
            means.append(_mean([columns[0][i] for i in picks]))
        else:
            means.append(_mean([columns[0][i] - columns[1][i] for i in picks]))
    means.sort()
    return means


def _interval(means: list[float], level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    low = means[int(tail * (len(means) - 1))]
    high = means[int((1 - tail) * (len(means) - 1))]
    return low, high


def confidence_interval(
    scores: Sequence[float], level: float = 0.95, resamples: int = RESAMPLES, seed: int = SEED
) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    return _interval(_resampled_means([scores], resamples, seed), level)


def paired_difference_interval(
    a: Sequence[float],
    b: Sequence[float],
    level: float = 0.95,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """Interval for mean(a - b) over the same queries. Containing 0 means no clear difference."""
    if len(a) != len(b):
        raise ValueError("paired scores must cover the same queries")
    if not a:
        return 0.0, 0.0
    return _interval(_resampled_means([a, b], resamples, seed), level)
