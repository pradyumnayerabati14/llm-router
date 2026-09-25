"""A small FastAPI service that routes a prompt to the strong or weak model.

    uvicorn llm_router.server:app --reload

POST /route            decide only, never calls an LLM (works with no API keys)
POST /v1/chat/completions   OpenAI-compatible: decide, then answer

The second endpoint calls a real provider only when OPENAI_API_KEY (or a
compatible base URL) is configured; otherwise it returns a clearly-labelled mock
answer so the service is demonstrable offline.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from llm_router.embeddings import embed
from llm_router.metrics import threshold_for_cost
from llm_router.routers import LogisticRouter

ROOT = pathlib.Path(__file__).resolve().parents[1]

STRONG_MODEL = os.getenv("STRONG_MODEL", "gpt-4o")
WEAK_MODEL = os.getenv("WEAK_MODEL", "mistral-small-latest")
DEFAULT_BUDGET = os.getenv("ROUTER_BUDGET", "30%")


def _default_alpha() -> float:
    """The cost dial: route to the strong model when score >= alpha.

    Default it to the threshold that sends ROUTER_BUDGET of evaluation traffic
    to the strong model, rather than a hard-coded 0.5. Scores are not calibrated
    probabilities - most prompts do not need the strong model, so they cluster
    low, and a fixed 0.5 would route nothing.
    """
    if "ROUTER_ALPHA" in os.environ:
        return float(os.environ["ROUTER_ALPHA"])
    metrics_path = ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        alpha = metrics.get("logistic", {}).get("alpha_for_budget", {}).get(DEFAULT_BUDGET)
        if alpha is not None:
            return float(alpha)
    return 0.5


ALPHA = _default_alpha()

app = FastAPI(title="LLM Router", version="0.1.0")
_router: LogisticRouter | None = None


def get_router() -> LogisticRouter:
    global _router
    if _router is None:
        _router = LogisticRouter.load(ROOT / "models" / "logistic.pt")
    return _router


@app.on_event("startup")
def warm_up() -> None:
    """Load the router and the embedding model before the first request.

    Without this the first call pays ~4 seconds of model loading, which looks
    like the router being slow when it is really a cold start.
    """
    get_router()
    embed(["warm up"])


class RouteRequest(BaseModel):
    prompt: str
    alpha: float | None = Field(default=None, description="override the cost dial")


class RouteResponse(BaseModel):
    prompt: str
    score: float
    alpha: float
    model: str
    tier: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "strong_model": STRONG_MODEL, "weak_model": WEAK_MODEL, "alpha": ALPHA}


@app.post("/route", response_model=RouteResponse)
def route_prompt(req: RouteRequest) -> RouteResponse:
    """Return the routing decision without calling any LLM."""
    start = time.perf_counter()
    alpha = ALPHA if req.alpha is None else req.alpha

    vector = embed([req.prompt])
    score = float(get_router().predict_proba(vector)[0])
    to_strong = score >= alpha

    return RouteResponse(
        prompt=req.prompt,
        score=round(score, 4),
        alpha=alpha,
        model=STRONG_MODEL if to_strong else WEAK_MODEL,
        tier="strong" if to_strong else "weak",
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "router"
    alpha: float | None = None


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict:
    """OpenAI-compatible endpoint: route first, then answer with the chosen model."""
    prompt = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    decision = route_prompt(RouteRequest(prompt=prompt, alpha=req.alpha))
    content = _complete(decision.model, req.messages)

    return {
        "id": f"router-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": decision.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "router": decision.model_dump(),
    }


def _complete(model: str, messages: list[ChatMessage]) -> str:
    """Call an OpenAI-compatible provider, or return a mock answer if unconfigured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            f"[mock answer - no OPENAI_API_KEY set] The router chose '{model}'. "
            "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) to get real completions."
        )

    import httpx

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [m.model_dump() for m in messages]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def calibrate(scores: np.ndarray, strong_fraction: float) -> float:
    """Convenience wrapper: pick alpha from a budget, e.g. "30% may go to GPT-4"."""
    return threshold_for_cost(scores, strong_fraction)
