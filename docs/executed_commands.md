# Reproduction command record

The remote non-interactive shell did not retain usable project command history.
The commands below are normalized, equivalent invocations reconstructed from
the launcher scripts, raw log names, `trainer_state.json`, `run_manifest.json`,
and the resulting artifacts. They intentionally use `$REPO` instead of a user
or host-specific project path and should not be treated as a byte-for-byte
terminal transcript.

## Shared environment

```bash
export REPO=/path/to/gated-attention-repro
cd "$REPO"
export HF_HOME=/mnt/ARD340/.hf_cache
export PATH="$REPO/.venv/bin:$PATH"
```

The physical worker had four L20 GPUs. The run manifests record a distributed
world size of two, so the equivalent launch command uses two processes.

## Validation

```bash
python -m unittest discover -s tests -v
python src/our/validate_gate.py
```

The final repository-wide test run completed 24 tests successfully.

## TinyStories smoke

```bash
set -o pipefail
LAUNCH="accelerate launch --num_processes 2" \
  bash scripts/run_smoke.sh 2>&1 | tee logs/smoke.log
```

This trained baseline, headwise, and elementwise with seed 20 for 2,015,232
processed tokens each, then analyzed nine snapshots per run and generated
`results/smoke/`.

## Phase A: official 1B checkpoints

```bash
set -o pipefail
bash scripts/run_eval_1b.sh 2>&1 | tee logs/phaseA_eval.log
```

This evaluated the three official 1B variants on the same FineWeb-Edu
validation data and generated `results/phaseA/`.

## Phase B: seed 20

```bash
set -o pipefail
SEEDS="20" \
VARIANTS="baseline headwise elementwise" \
LAUNCH="accelerate launch --num_processes 2" \
  bash scripts/run_train.sh 2>&1 | tee logs/phaseB.log
```

## Phase B: additional seeds

```bash
set -o pipefail
SEEDS="21 22" \
VARIANTS="baseline elementwise" \
LAUNCH="accelerate launch --num_processes 2" \
  bash scripts/run_train.sh 2>&1 | tee logs/phaseB_seeds21_22.log
```

The two commands above produced seven token-matched formal runs. Every run
reached 500,006,912 planned processed tokens and generated all nine analysis
snapshots.

## Recovered aggregation and final checks

The additional-seed launcher reached its final shell aggregation command after
all four requested training and snapshot-analysis runs had completed, but that
shell no longer resolved the bare `python` executable. Aggregation was rerun
with the virtual-environment interpreter:

```bash
HF_HOME=/mnt/ARD340/.hf_cache \
  .venv/bin/python src/our/compare_runs.py \
  --out_root outputs/fineweb_edu_500m \
  --out results/phaseB

HF_HOME=/mnt/ARD340/.hf_cache \
  .venv/bin/python -m unittest discover -s tests -v
```

The regenerated Phase B summary contains all seven runs. The test suite passed
24/24 tests after the final analysis change.
