"""Statistical utilities for Phase 5: bootstrap CIs and paired tests.

All functions operate on lists of numeric results. Used to compare the graph
monitor against sequence baselines on lead time / normalized detection time
at matched FPR.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as _stats

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy optional
    _HAS_SCIPY = False


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    stat: str = "mean",
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Args:
        values: sample values.
        n_bootstrap: number of bootstrap resamples.
        alpha: significance level (CI = 100*(1-alpha)%).
        seed: random seed for reproducibility.
        stat: "mean" or "median".

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(seed)

    if stat == "mean":
        point = float(values.mean())
    elif stat == "median":
        point = float(np.median(values))
    else:
        raise ValueError(f"Unknown stat: {stat}")

    if n == 1:
        return point, point, point

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        if stat == "mean":
            bootstrap_stats.append(float(sample.mean()))
        else:
            bootstrap_stats.append(float(np.median(sample)))

    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


def ci_for_difference(
    graph_values: List[float],
    baseline_values: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the paired difference (graph - baseline).

    Resamples pairs together to preserve pairing.
    """
    g = np.asarray(graph_values, dtype=float)
    b = np.asarray(baseline_values, dtype=float)
    n = min(len(g), len(b))
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    g = g[:n]
    b = b[:n]
    diffs = g - b
    point = float(diffs.mean())

    rng = np.random.RandomState(seed)
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_diffs.append(float(diffs[idx].mean()))

    lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))
    return point, lower, upper


def paired_ttest(
    graph_values: List[float], baseline_values: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Paired t-test: graph vs baseline.

    Returns (statistic, p_value). Returns (None, None) if scipy unavailable
    or insufficient data.
    """
    if not _HAS_SCIPY:
        return None, None
    g = np.asarray(graph_values, dtype=float)
    b = np.asarray(baseline_values, dtype=float)
    n = min(len(g), len(b))
    if n < 3:
        return None, None
    stat, p = _stats.ttest_rel(g[:n], b[:n])
    return float(stat), float(p)


def wilcoxon_signed_rank(
    graph_values: List[float], baseline_values: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Wilcoxon signed-rank test: graph vs baseline (non-parametric paired).

    Returns (statistic, p_value). Returns (None, None) if scipy unavailable
    or insufficient data.
    """
    if not _HAS_SCIPY:
        return None, None
    g = np.asarray(graph_values, dtype=float)
    b = np.asarray(baseline_values, dtype=float)
    n = min(len(g), len(b))
    if n < 3:
        return None, None
    try:
        stat, p = _stats.wilcoxon(g[:n] - b[:n])
    except ValueError:  # all differences zero
        return 0.0, 1.0
    return float(stat), float(p)


def detect_rate(values: List[bool]) -> float:
    """Fraction of True values."""
    if not values:
        return 0.0
    return float(sum(1 for v in values if v)) / len(values)
