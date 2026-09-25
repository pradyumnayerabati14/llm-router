"""Tests for the label logic - the part most likely to be silently wrong."""

from __future__ import annotations

import json

import pandas as pd

from llm_router.data import SCORE_TO_Y, arena_to_pairs, judge_to_pairs, parse_prompt


def _battle(model_a: str, model_b: str, winner: str, prompt: str) -> dict:
    return {
        "model_a": model_a,
        "model_b": model_b,
        "prompt": json.dumps([prompt]),
        "winner_model_a": int(winner == "a"),
        "winner_model_b": int(winner == "b"),
        "winner_tie": int(winner == "tie"),
    }


def test_label_is_flipped_when_strong_model_is_b():
    """The strong model wins both battles, so both rows must be y = 1.0.

    This is the bug that would never crash: in the second battle the strong
    model is model_b, so winner_model_b - not winner_model_a - means the strong
    model won. Getting it wrong teaches the router the exact opposite.
    """
    df = pd.DataFrame(
        [
            _battle(
                "gpt-4-1106-preview", "mixtral-8x7b-instruct-v0.1", "a", "explain gradient descent"
            ),
            _battle(
                "mixtral-8x7b-instruct-v0.1", "gpt-4-1106-preview", "b", "explain backpropagation"
            ),
        ]
    )
    out = arena_to_pairs(df)
    assert len(out) == 2
    assert list(out.y) == [1.0, 1.0]


def test_weak_win_and_tie_labels():
    df = pd.DataFrame(
        [
            _battle("gpt-4-1106-preview", "claude-2.0", "b", "what is the capital of France?"),
            _battle("claude-2.0", "gpt-4-0314", "tie", "write a haiku about coffee"),
        ]
    )
    out = arena_to_pairs(df)
    assert list(out.y) == [0.0, 0.5]


def test_same_tier_battles_are_dropped():
    """Strong vs strong and weak vs weak say nothing about strong-vs-weak."""
    df = pd.DataFrame(
        [
            _battle("gpt-4-0314", "gpt-4-0613", "a", "summarise this article for me"),
            _battle("claude-2.0", "gemini-pro", "b", "summarise this article for me"),
            _battle("gpt-4-0314", "llama-13b", "a", "unknown tier model should drop"),
        ]
    )
    assert len(arena_to_pairs(df)) == 0


def test_parse_prompt_keeps_first_turn_only():
    turns = ["explain gradient descent", "and now explain momentum"]
    assert parse_prompt(json.dumps(turns)) == "explain gradient descent"


def test_parse_prompt_rejects_short_and_null():
    assert parse_prompt(json.dumps(["hi"])) is None  # under 16 characters
    assert parse_prompt(json.dumps([None])) is None
    assert parse_prompt(None) is None


def test_judge_scores_map_to_targets():
    df = pd.DataFrame(
        {
            "prompt": ["a prompt long enough to keep"] * 5,
            "mixtral_score": [1, 2, 3, 4, 5],
        }
    )
    out = judge_to_pairs(df.assign(prompt=[f"prompt number {i} padded" for i in range(5)]))
    assert list(out.y) == [SCORE_TO_Y[s] for s in [1, 2, 3, 4, 5]]
    assert list(out.y) == [1.0, 1.0, 1.0, 0.5, 0.0]
