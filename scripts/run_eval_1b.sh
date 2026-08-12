#!/usr/bin/env bash
# Phase A: evaluate the three released 1B checkpoints on one validation split.
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
cd "$PROJ"

MODELS_DIR="${MODELS_DIR:-./models_1b}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-./data/fineweb_edu_eval}"
OUT="${OUT:-./results/phaseA}"
VARIANTS="${VARIANTS:-baseline gate_headwise gate_elementwise}"

if [ ! -f "$EVAL_DATA_DIR/meta.json" ]; then
  python src/our/prep_data.py \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config default \
    --tokenizer Qwen/Qwen3-0.6B \
    --block_size 2048 \
    --train_tokens 2048000 \
    --validation_tokens 2048000 \
    --streaming \
    --overwrite \
    --output_dir "$EVAL_DATA_DIR"
fi

if [ ! -d "$MODELS_DIR/1B_baseline" ]; then
  echo "[phaseA] downloading official 1B checkpoints..."
  if command -v hf >/dev/null 2>&1; then
    hf download QwQZh/gated_attention --include "1B_*/*" --local-dir "$MODELS_DIR"
  else
    huggingface-cli download QwQZh/gated_attention \
      --include "1B_*/*" --local-dir "$MODELS_DIR"
  fi
fi

for variant in $VARIANTS; do
  python src/our/eval_attention.py \
    --model_path "$MODELS_DIR/1B_$variant" \
    --variant "$variant" \
    --output_dir "$OUT" \
    --eval_data_dir "$EVAL_DATA_DIR" \
    --ppl_context_length 1024 \
    --ppl_max_tokens 262144 \
    --sink_context_length 512 \
    --sink_samples 16
done

python src/our/aggregate_results.py --results_dir "$OUT"
echo "[phaseA] DONE -> $OUT"
