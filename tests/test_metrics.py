"""Tests for the paper's metrics (section 3.2)."""

from __future__ import annotations

import numpy as np

from llm_router.metrics import apgr, cpt, pgr, route, router_quality, threshold_for_cost


def test_pgr_endpoints():
    assert pgr(router_q=0.4, strong_q=1.0, weak_q=0.4) == 0.0  # weak model alone
    assert pgr(router_q=1.0, strong_q=1.0, weak_q=0.4) == 1.0  # strong model alone
    assert 0.49 < pgr(router_q=0.7, strong_q=1.0, weak_q=0.4) < 0.51


def test_threshold_matches_requested_budget():
    scores = np.linspace(0, 1, 1000)
    alpha = threshold_for_cost(scores, 0.30)
    assert 0.29 < route(scores, alpha).mean() < 0.31


def test_random_router_recovers_gap_proportionally():
    """A random router's PGR should track its cost: spend 50%, recover ~50%."""
    rng = np.random.default_rng(0)
    n = 5000
    weak_q, strong_q = rng.random(n), np.ones(n)
    scores = rng.random(n)

    alpha = threshold_for_cost(scores, 0.5)
    quality = router_quality(route(scores, alpha), strong_q, weak_q)
    assert 0.45 < pgr(quality, strong_q.mean(), weak_q.mean()) < 0.55


def test_oracle_beats_random():
    rng = np.random.default_rng(0)
    n = 5000
    weak_q, strong_q = rng.random(n), np.ones(n)
    oracle = 1 - weak_q  # prioritise queries where the weak model is worst

    assert apgr(oracle, strong_q, weak_q) > apgr(rng.random(n), strong_q, weak_q)
    assert cpt(oracle, strong_q, weak_q, 0.5) < cpt(rng.random(n), strong_q, weak_q, 0.5)
