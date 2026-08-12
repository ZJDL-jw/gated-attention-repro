#!/usr/bin/env bash
# Formal Phase B: FineWeb-Edu, three variants, token-matched training.
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
cd "$PROJ"

LAUNCH="${LAUNCH:-python}"
CFG="${CFG:-./configs/qwen3_tiny.json}"
DATA_DIR="${DATA_DIR:-./data/fineweb_edu_500m}"
OUT_ROOT="${OUT_ROOT:-./outputs/fineweb_edu_500m}"
TRAIN_TOKENS="${TRAIN_TOKENS:-500000000}"
VALIDATION_TOKENS="${VALIDATION_TOKENS:-8000000}"
TARGET_TOKENS="${TARGET_TOKENS:-500000000}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-default}"
VARIANTS="${VARIANTS:-baseline headwise elementwise}"
SEEDS="${SEEDS:-20}"

if [ ! -f "$DATA_DIR/meta.json" ]; then
  python src/our/prep_data.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG" \
    --tokenizer Qwen/Qwen3-0.6B \
    --block_size 2048 \
    --train_tokens "$TRAIN_TOKENS" \
    --validation_tokens "$VALIDATION_TOKENS" \
    --streaming \
    --overwrite \
    --output_dir "$DATA_DIR"
fi

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    run_dir="$OUT_ROOT/$variant/seed-$seed"
    echo "[formal] variant=$variant seed=$seed -> $run_dir"
    $LAUNCH src/our/train.py \
      --variant "$variant" \
      --config "$CFG" \
      --data_dir "$DATA_DIR" \
      --output_dir "$run_dir" \
      --target_train_tokens "$TARGET_TOKENS" \
      --seed "$seed" \
      --analysis_context_lengths 512,1024,2048,4096
  done
done

python src/our/compare_runs.py --out_root "$OUT_ROOT" --out ./results/phaseB
echo "[formal] DONE -> $OUT_ROOT and ./results/phaseB"
