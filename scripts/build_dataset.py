"""Build the training/validation tables from the raw parquet downloads.

    python scripts/build_dataset.py

Reads   data/arena_55k.parquet, data/gpt4_judge_train.parquet,
        data/gpt4_judge_val.parquet
Writes  data/train.parquet, data/eval.parquet
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llm_router.data import (
    MODEL_TIERS,
    arena_to_pairs,
    judge_to_eval,
    judge_to_pairs,
)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
JUDGE_TRAIN_SAMPLE = 40_000  # keep the run laptop-sized; arena is kept in full
SEED = 0


def main() -> None:
    arena = pd.read_parquet(DATA / "arena_55k.parquet")
    known = set(MODEL_TIERS)
    seen = set(arena.model_a) | set(arena.model_b)
    missing = sorted(seen - known)
    print(f"arena battles: {len(arena):,}  models seen: {len(seen)}")
    print(f"models missing from the tier table ({len(missing)}): {missing}\n")

    arena_pairs = arena_to_pairs(arena)
    print(f"arena -> strong-vs-weak pairs: {len(arena_pairs):,}")

    judge = pd.read_parquet(DATA / "gpt4_judge_train.parquet")
    judge_pairs = judge_to_pairs(judge)
    if len(judge_pairs) > JUDGE_TRAIN_SAMPLE:
        judge_pairs = judge_pairs.sample(JUDGE_TRAIN_SAMPLE, random_state=SEED)
    print(f"judge -> pairs (sampled): {len(judge_pairs):,}")

    train = (
        pd.concat([arena_pairs, judge_pairs], ignore_index=True)
        .drop_duplicates(subset="prompt")
        .sample(frac=1.0, random_state=SEED)  # shuffle so sources interleave
        .reset_index(drop=True)
    )

    val_raw = pd.read_parquet(DATA / "gpt4_judge_val.parquet")
    evalset = judge_to_eval(val_raw)
    # The judge validation split is held out by the dataset authors, but make
    # sure no prompt leaks across: a router that memorises prompts would score
    # far too well.
    evalset = evalset[~evalset.prompt.isin(set(train.prompt))].reset_index(drop=True)

    train.to_parquet(DATA / "train.parquet", index=False)
    evalset.to_parquet(DATA / "eval.parquet", index=False)

    print(f"\ntrain rows: {len(train):,}")
    print(train.groupby("source").agg(rows=("y", "size"), mean_y=("y", "mean")))
    print(f"\neval rows: {len(evalset):,}")
    print(f"mean weak quality on eval: {evalset.weak_quality.mean():.3f}")


if __name__ == "__main__":
    main()
