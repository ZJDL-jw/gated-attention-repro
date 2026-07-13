#!/usr/bin/env bash
# Phase A on the 4xL20 machine.
# 1) downloads the released 1B models if missing
# 2) runs eval_attention.py for each of the three variants
# 3) aggregates results into a comparison table + chart
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
cd "$PROJ"

MODELS_DIR="${MODELS_DIR:-./models_1b}"
OUT="${OUT:-./results/phaseA}"
VARIANTS="baseline gate_headwise gate_elementwise"

# 1) fetch the official 1B models (each ~2GB; needs ~6GB free)
if [ ! -d "$MODELS_DIR/1B_baseline" ]; then
  echo "[run_eval_1b] downloading 1B models from HF (QwQZh/gated_attention)..."
  huggingface-cli download QwQZh/gated_attention --include "1B_*/*" --local-dir "$MODELS_DIR"
fi

# 2) evaluate each variant
for v in $VARIANTS; do
  echo "[run_eval_1b] evaluating $v ..."
  python src/our/eval_attention.py \
    --model_path "$MODELS_DIR/1B_$v" \
    --variant "$v" \
    --output_dir "$OUT" \
    --prompt "Sparse gating mechanism mitigates attention sink." \
    --ppl_text ./data/ppl_sample.txt
done

# 3) aggregate
python src/our/aggregate_results.py --results_dir "$OUT"
echo "[run_eval_1b] DONE. Results in $OUT"
