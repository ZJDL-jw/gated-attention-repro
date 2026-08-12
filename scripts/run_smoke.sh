#!/usr/bin/env bash
# Cheap end-to-end smoke run. Override TARGET_TOKENS=20000000 for a fuller run.
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
cd "$PROJ"

LAUNCH="${LAUNCH:-python}"
DATA_DIR="${DATA_DIR:-./data/tinystories_20m}"
OUT_ROOT="${OUT_ROOT:-./outputs/smoke}"
TARGET_TOKENS="${TARGET_TOKENS:-2000000}"
VARIANTS="${VARIANTS:-baseline headwise elementwise}"

python src/our/validate_gate.py

if [ ! -f "$DATA_DIR/meta.json" ]; then
  python src/our/prep_data.py \
    --dataset_name roneneldan/TinyStories \
    --tokenizer Qwen/Qwen3-0.6B \
    --block_size 2048 \
    --train_tokens 20000000 \
    --validation_tokens 2000000 \
    --streaming \
    --overwrite \
    --output_dir "$DATA_DIR"
fi

for variant in $VARIANTS; do
  run_dir="$OUT_ROOT/$variant/seed-20"
  echo "[smoke] variant=$variant -> $run_dir"
  $LAUNCH src/our/train.py \
    --variant "$variant" \
    --config ./configs/qwen3_tiny.json \
    --data_dir "$DATA_DIR" \
    --output_dir "$run_dir" \
    --target_train_tokens "$TARGET_TOKENS" \
    --trainer_eval_tokens 131072 \
    --analysis_ppl_tokens 32768 \
    --analysis_sink_samples 2 \
    --seed 20
done

python src/our/compare_runs.py --out_root "$OUT_ROOT" --out ./results/smoke
echo "[smoke] DONE -> $OUT_ROOT and ./results/smoke"
