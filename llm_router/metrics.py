"""The paper's evaluation metrics (section 3.2).

Accuracy is the wrong metric for a router. About 86% of prompts in the judge
data do not need the strong model, so "always use the weak model" already looks
86% accurate while quietly failing every hard query. What matters is the
trade-off: how much of the quality gap do we recover, for how many expensive
calls?

    PGR   - performance gap recovered (eq. 6). 0 = as bad as the weak model
            alone, 1 = as good as the strong model alone.
    APGR  - average PGR across all cost levels (eq. 7/8). One number per router.
    CPT(x)- call-performance threshold (section 3.2): the smallest share of
            strong-model calls needed to reach PGR = x. Lower is better.
"""

from __future__ import annotations

import numpy as np


def threshold_for_cost(scores: np.ndarray, strong_fraction: float) -> float:
    """The alpha that sends exactly ``strong_fraction`` of queries to the strong model.

    This is how alpha is chosen in practice: you do not pick a probability out
    of the air, you pick a budget ("30% of traffic may go to GPT-4") and read
    off the matching quantile of the router's scores.
    """
    if strong_fraction <= 0:
        return np.inf  # nothing clears the bar
    if strong_fraction >= 1:
        return -np.inf  # everything clears it
    return float(np.quantile(scores, 1.0 - strong_fraction))


def route(scores: np.ndarray, alpha: float) -> np.ndarray:
    """True = send to the strong model. Equation 2."""
    return scores >= alpha


def router_quality(
    to_strong: np.ndarray, strong_quality: np.ndarray, weak_quality: np.ndarray
) -> float:
    """Average response quality once each query is answered by its chosen model."""
    return float(np.where(to_strong, strong_quality, weak_quality).mean())


def pgr(router_q: float, strong_q: float, weak_q: float) -> float:
    """Performance gap recovered, equation 6."""
    gap = strong_q - weak_q
    if abs(gap) < 1e-12:
        return 1.0  # the two models are equally good; nothing to recover
    return (router_q - weak_q) / gap


def call_performance_curve(
    scores: np.ndarray,
    strong_quality: np.ndarray,
    weak_quality: np.ndarray,
    fractions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PGR at each cost level: the curve plotted in figure 1 of the paper."""
    if fractions is None:
        fractions = np.linspace(0.0, 1.0, 11)
    strong_mean, weak_mean = float(strong_quality.mean()), float(weak_quality.mean())

    pgrs = []
    for frac in fractions:
        alpha = threshold_for_cost(scores, float(frac))
        quality = router_quality(route(scores, alpha), strong_quality, weak_quality)
        pgrs.append(pgr(quality, strong_mean, weak_mean))
    return np.asarray(fractions), np.asarray(pgrs)


def apgr(scores: np.ndarray, strong_quality: np.ndarray, weak_quality: np.ndarray) -> float:
    """Average PGR over ten evenly spaced cost levels, equation 8."""
    fractions = np.linspace(0.1, 1.0, 10)
    _, pgrs = call_performance_curve(scores, strong_quality, weak_quality, fractions)
    return float(pgrs.mean())


def cpt(
    scores: np.ndarray,
    strong_quality: np.ndarray,
    weak_quality: np.ndarray,
    target_pgr: float,
    resolution: int = 101,
) -> float:
    """Smallest % of strong-model calls that reaches ``target_pgr``. NaN if never."""
    fractions = np.linspace(0.0, 1.0, resolution)
    _, pgrs = call_performance_curve(scores, strong_quality, weak_quality, fractions)
    hits = np.where(pgrs >= target_pgr)[0]
    return float(fractions[hits[0]] * 100) if len(hits) else float("nan")
