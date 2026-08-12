#!/usr/bin/env bash
# Re-run snapshot analysis without retraining.
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
cd "$PROJ"

DATA_DIR="${DATA_DIR:-./data/fineweb_edu_500m}"
OUT_ROOT="${OUT_ROOT:-./outputs/fineweb_edu_500m}"
VARIANTS="${VARIANTS:-baseline headwise elementwise}"
SEEDS="${SEEDS:-20}"

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    python src/our/analyze_checkpoints.py \
      --run_dir "$OUT_ROOT/$variant/seed-$seed" \
      --data_dir "$DATA_DIR" \
      --variant "$variant" \
      --context_lengths 512,1024,2048,4096
  done
done

python src/our/compare_runs.py --out_root "$OUT_ROOT" --out ./results/phaseB
