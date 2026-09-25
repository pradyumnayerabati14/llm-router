"""Route a handful of example prompts and show the decisions.

    python scripts/demo.py

Prints one line per prompt: the router's score, the chosen model, and the
latency of the decision itself. No LLM is called, so this runs with no API keys.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm_router.embeddings import embed
from llm_router.routers import LogisticRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUDGET = "30%"  # share of traffic we are willing to send to the strong model

PROMPTS = [
    "What is the capital of France?",
    "Translate 'good morning' into Spanish.",
    "Write a haiku about coffee.",
    "Summarise the following in one sentence: the meeting was postponed to Friday.",
    "Prove that there are infinitely many primes of the form 4k+3.",
    "Here is a 300-line Python traceback about a circular import in a Django app. Diagnose the root cause and propose a fix.",
    "Design a rate limiter for a distributed API with per-tenant quotas, and explain the trade-offs.",
    "Derive the gradient of the softmax cross-entropy loss with respect to the logits.",
]


def main() -> None:
    router = LogisticRouter.load(ROOT / "models" / "logistic.pt")

    # alpha comes from a cost budget, not from a guess. A raw probability of 0.5
    # is meaningless here: most training prompts did not need the strong model,
    # so the scores sit well below 0.5 and a fixed 0.5 would route nothing.
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    alpha = metrics["logistic"]["alpha_for_budget"][BUDGET]

    embed(["warm up the embedding model"])  # exclude model load from timings
    t0 = time.perf_counter()
    vectors = embed(PROMPTS)
    scores = router.predict_proba(vectors)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"budget = {BUDGET} of traffic to the strong model  ->  alpha = {alpha}\n")
    print(f"{'score':>6}  {'route to':<8}  prompt")
    print("-" * 100)
    for prompt, score in zip(PROMPTS, scores):
        target = "STRONG" if score >= alpha else "weak"
        text = prompt if len(prompt) <= 76 else prompt[:73] + "..."
        print(f"{score:6.3f}  {target:<8}  {text}")

    strong_share = (scores >= alpha).mean() * 100
    print(
        f"\n{strong_share:.0f}% routed to the strong model | "
        f"{elapsed_ms:.0f} ms for {len(PROMPTS)} decisions "
        f"({elapsed_ms / len(PROMPTS):.1f} ms each, embedding included)"
    )


if __name__ == "__main__":
    main()
