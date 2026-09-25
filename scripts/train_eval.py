"""Train every router, evaluate them on the held-out split, save results.

    python scripts/train_eval.py

Writes  models/logistic.pt, results/metrics.json, results/call_performance.png
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm_router.embeddings import embed
from llm_router.metrics import (
    apgr,
    call_performance_curve,
    cpt,
    threshold_for_cost,
)
from llm_router.routers import (
    LogisticRouter,
    RandomRouter,
    SimilarityWeightedRouter,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, MODELS, RESULTS = ROOT / "data", ROOT / "models", ROOT / "results"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    train = pd.read_parquet(DATA / "train.parquet")
    evalset = pd.read_parquet(DATA / "eval.parquet")
    print(f"train {len(train):,} rows | eval {len(evalset):,} rows")

    X_train = embed(train.prompt.tolist(), cache_key="train")
    X_eval = embed(evalset.prompt.tolist(), cache_key="eval")
    y_train = train.y.to_numpy(dtype=np.float32)

    strong_q = evalset.strong_quality.to_numpy(dtype=np.float32)
    weak_q = evalset.weak_quality.to_numpy(dtype=np.float32)
    print(f"\nweak model alone: {weak_q.mean():.3f} | strong model alone: {strong_q.mean():.3f}")

    routers = [RandomRouter(seed=0), SimilarityWeightedRouter(), LogisticRouter()]
    results = {}

    for router in routers:
        t0 = time.perf_counter()
        router.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = router.predict_proba(X_eval)
        predict_s = time.perf_counter() - t0

        results[router.name] = {
            "apgr": apgr(scores, strong_q, weak_q),
            "cpt_50": cpt(scores, strong_q, weak_q, 0.5),
            "cpt_80": cpt(scores, strong_q, weak_q, 0.8),
            "fit_seconds": round(fit_s, 2),
            "predict_seconds": round(predict_s, 2),
            "queries_per_second": round(len(X_eval) / predict_s, 1),
            # A raw probability is not a budget. These are the alphas that send
            # 20% / 30% / 50% of this traffic to the strong model.
            "alpha_for_budget": {
                f"{int(f * 100)}%": round(threshold_for_cost(scores, f), 4) for f in (0.2, 0.3, 0.5)
            },
            "scores": scores.tolist(),
        }
        r = results[router.name]
        print(
            f"{router.name:<12} APGR {r['apgr']:.3f}  CPT(50%) {r['cpt_50']:.1f}%  "
            f"CPT(80%) {r['cpt_80']:.1f}%  {r['queries_per_second']:.0f} q/s"
        )
        if isinstance(router, LogisticRouter):
            router.save(MODELS / "logistic.pt")

    plot(results, strong_q, weak_q)

    summary = {
        name: {k: v for k, v in vals.items() if k != "scores"} for name, vals in results.items()
    }
    summary["_meta"] = {
        "train_rows": len(train),
        "eval_rows": len(evalset),
        "weak_quality_mean": float(weak_q.mean()),
        "strong_quality_mean": float(strong_q.mean()),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {RESULTS / 'metrics.json'} and {RESULTS / 'call_performance.png'}")


def plot(results: dict, strong_q: np.ndarray, weak_q: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, vals in results.items():
        fracs, pgrs = call_performance_curve(
            np.asarray(vals["scores"], dtype=np.float32), strong_q, weak_q
        )
        ax.plot(fracs * 100, pgrs, marker="o", ms=3, label=f"{name} (APGR {vals['apgr']:.3f})")

    ax.set_xlabel("% of queries sent to the strong model (cost)")
    ax.set_ylabel("performance gap recovered (PGR)")
    ax.set_title("Call-performance trade-off on held-out judge data")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "call_performance.png", dpi=150)


if __name__ == "__main__":
    main()
