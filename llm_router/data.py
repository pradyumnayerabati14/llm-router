"""Turn raw preference datasets into a single (prompt, y) training table.

Two sources, as in RouteLLM (Ong et al., ICLR 2025) section 4.1:

* ``D_arena``  - real Chatbot Arena battles, labelled by the human who asked.
* ``D_judge``  - Nectar prompts where GPT-4 graded Mixtral's answer 1-5.

Both are reduced to one target ``y = P(the strong model is needed)``:

    y = 1.0   the strong model's answer was better
    y = 0.5   the two were equivalent
    y = 0.0   the weak model was good enough

Everything downstream (routers, metrics) only ever sees ``prompt`` and ``y``.
"""

from __future__ import annotations

import json

import pandas as pd

# --- Model tiers -----------------------------------------------------------
# Appendix A of the paper. Models are grouped into quality tiers so that we can
# learn "strong tier vs weak tier" instead of "GPT-4 vs Mixtral": any specific
# model pair covers <0.1% of battles, which is far too sparse to train on.
MODEL_TIERS: dict[str, int] = {
    # Tier 0
    "gpt-4-0125-preview": 0,
    "gpt-4-1106-preview": 0,
    # Tier 1
    "gpt-4-0314": 1,
    "gpt-4-0613": 1,
    "mistral-medium": 1,
    "claude-1": 1,
    "qwen1.5-72b-chat": 1,
    # Tier 2
    "claude-2.0": 2,
    "mixtral-8x7b-instruct-v0.1": 2,
    "claude-2.1": 2,
    "gemini-pro-dev-api": 2,
    "gpt-3.5-turbo-0314": 2,
    "gpt-3.5-turbo-0613": 2,
    "gemini-pro": 2,
    "gpt-3.5-turbo-0125": 2,
    "claude-instant-1": 2,
    "yi-34b-chat": 2,
    "starling-lm-7b-alpha": 2,
    "wizardlm-70b": 2,
    "vicuna-33b": 2,
    "tulu-2-dpo-70b": 2,
    "nous-hermes-2-mixtral-8x7b-dpo": 2,
    "llama-2-70b-chat": 2,
    "openchat-3.5": 2,
    # Tier 3 (unused for labels, kept so tier lookups are explicit)
    "llama2-70b-steerlm-chat": 3,
    "pplx-70b-online": 3,
    "dolphin-2.2.1-mistral-7b": 3,
    "gpt-3.5-turbo-1106": 3,
    "deepseek-llm-67b-chat": 3,
    "openhermes-2.5-mistral-7b": 3,
    "openchat-3.5-0106": 3,
    "wizardlm-13b": 3,
    "mistral-7b-instruct-v0.2": 3,
    "solar-10.7b-instruct-v1.0": 3,
    "zephyr-7b-beta": 3,
    "zephyr-7b-alpha": 3,
    "codellama-34b-instruct": 3,
    "mpt-30b-chat": 3,
    "llama-2-13b-chat": 3,
    "vicuna-13b": 3,
    "qwen1.5-7b-chat": 3,
    "pplx-7b-online": 3,
    "falcon-180b-chat": 3,
    "llama-2-7b-chat": 3,
    "guanaco-33b": 3,
    "qwen-14b-chat": 3,
}

STRONG_TIERS = {0, 1}  # top two tiers -> "strong" class
WEAK_TIERS = {2}  # third tier      -> "weak" class

# GPT-4 judged Mixtral's answer against GPT-4's own answer, so the score is a
# comparison, not an absolute grade. 1-3 = clearly worse, 4 = close, 5 = equal.
SCORE_TO_Y: dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0}

MIN_PROMPT_CHARS = 16  # section 5: prompts shorter than this are dropped


def parse_prompt(raw: str | None) -> str | None:
    """Return the first user turn of an Arena prompt, or None if unusable.

    Arena stores prompts as a JSON list of turns, e.g. ``'["hi", "and then?"]'``.
    We keep only the first turn: at serving time the router must decide before
    any model has replied, so a first turn is all it would ever have.
    """
    if not isinstance(raw, str):
        return None
    try:
        turns = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        turns = [raw]
    if isinstance(turns, str):
        turns = [turns]
    if not turns:
        return None
    first = turns[0]
    if not isinstance(first, str) or len(first.strip()) < MIN_PROMPT_CHARS:
        return None
    return first.strip()


def _tier(model: str) -> int | None:
    return MODEL_TIERS.get(model)


def arena_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Battles -> (prompt, y) rows, keeping only strong-tier vs weak-tier ones."""
    rows = []
    for row in df.itertuples(index=False):
        tier_a, tier_b = _tier(row.model_a), _tier(row.model_b)
        if tier_a is None or tier_b is None:
            continue

        # Which side is the strong model? Skip battles inside the same class.
        if tier_a in STRONG_TIERS and tier_b in WEAK_TIERS:
            strong_is_a = True
        elif tier_b in STRONG_TIERS and tier_a in WEAK_TIERS:
            strong_is_a = False
        else:
            continue

        prompt = parse_prompt(row.prompt)
        if prompt is None:
            continue

        # y is always from the STRONG model's point of view. Forgetting to flip
        # when the strong model is model_b teaches the router the exact opposite
        # of the truth, and nothing would crash to tell you.
        if row.winner_tie:
            y = 0.5
        elif row.winner_model_a:
            y = 1.0 if strong_is_a else 0.0
        else:
            y = 0.0 if strong_is_a else 1.0

        rows.append((prompt, y))

    return pd.DataFrame(rows, columns=["prompt", "y"]).assign(source="arena")


def judge_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """GPT-4-judge rows -> (prompt, y). Nectar prompts are already plain text."""
    out = df.loc[:, ["prompt", "mixtral_score"]].copy()
    out["prompt"] = out["prompt"].map(
        lambda p: p.strip() if isinstance(p, str) and len(p.strip()) >= MIN_PROMPT_CHARS else None
    )
    out = out.dropna(subset=["prompt"])
    out["y"] = out["mixtral_score"].map(SCORE_TO_Y)
    out = out.dropna(subset=["y"])
    return out[["prompt", "y"]].assign(source="judge")


def judge_to_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Held-out rows for evaluation, with a per-prompt quality for each model.

    The paper scores routers on benchmarks where every question has a known
    correctness for both models. We have no such benchmark locally, so we use
    the judge's 1-5 score as the weak model's quality on that prompt, rescaled
    to [0, 1], and treat the strong model (the judge's reference answer) as 1.0.
    """
    out = df.loc[:, ["prompt", "mixtral_score"]].copy()
    out["prompt"] = out["prompt"].map(
        lambda p: p.strip() if isinstance(p, str) and len(p.strip()) >= MIN_PROMPT_CHARS else None
    )
    out = out.dropna(subset=["prompt"])
    out["weak_quality"] = (out["mixtral_score"] - 1) / 4.0
    out["strong_quality"] = 1.0
    return out[["prompt", "weak_quality", "strong_quality", "mixtral_score"]]
