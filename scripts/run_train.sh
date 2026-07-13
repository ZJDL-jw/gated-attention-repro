#!/usr/bin/env bash
# Phase B on the 4xL20 machine.
#   1) prepares a small tokenized corpus if missing
#   2) trains baseline / headwise / elementwise (same config, same data)
#   3) each run saves a checkpoint + sink curve + attention maps
#
# Usage:
#   PROJ=$(pwd) LAUNCH="accelerate launch --num_processes 4" \
#     bash scripts/run_train.sh
#
# If LAUNCH is omitted, training falls back to a single GPU / CPU.
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
cd "$PROJ"

LAUNCH="${LAUNCH:-python}"
CFG="${CFG:-./configs/qwen3_tiny.json}"
DATA_DIR="${DATA_DIR:-./data/tinystories_50M}"
OUT_ROOT="${OUT_ROOT:-./outputs}"
MAX_TOKENS="${MAX_TOKENS:-50000000}"
MAX_STEPS="${MAX_STEPS:-3000}"
VARIANTS="baseline headwise elementwise"

# 1) data
if [ ! -d "$DATA_DIR" ]; then
  echo "[run_train] preparing corpus ($MAX_TOKENS tokens) -> $DATA_DIR"
  python src/our/prep_data.py \
    --dataset_name roneneldan/TinyStories \
    --tokenizer Qwen/Qwen3-0.6B \
    --block_size 2048 \
    --max_tokens "$MAX_TOKENS" \
    --output_dir "$DATA_DIR"
fi

# 2) train each variant
for v in $VARIANTS; do
  echo "[run_train] ===== training variant: $v ====="
  $LAUNCH src/our/train.py \
    --variant "$v" \
    --config "$CFG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_ROOT/$v" \
    --max_steps "$MAX_STEPS"
done

echo "[run_train] DONE. Checkpoints + sink curves + maps under $OUT_ROOT/"
echo "[run_train] Next: compare trained variants with src/our/compare_runs.py"
