#!/usr/bin/env bash
# Download the raw preference datasets (~265MB) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
base="https://huggingface.co/datasets"

curl -L -o data/arena_55k.parquet \
  "$base/lmarena-ai/arena-human-preference-55k/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
curl -L -o data/gpt4_judge_train.parquet \
  "$base/routellm/gpt4_dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
curl -L -o data/gpt4_judge_val.parquet \
  "$base/routellm/gpt4_dataset/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"

ls -lh data/
