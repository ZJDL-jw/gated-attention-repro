# Reproducing *Gated Attention for Large Language Models* (NeurIPS 2025)

A from-scratch reproduction of **Gated Attention** — a query-dependent sigmoid
gate inserted *after* the softmax attention output (before `o_proj`) that
introduces (1) non-linearity breaking the low-rank `Wv→Wo` bottleneck and
(2) input-dependent sparsity that mitigates the attention-sink effect.

- Paper: *Gated Attention for Large Language Models: Non-linearity, Sparsity,
  and Attention-Sink-Free* (Qiu et al., NeurIPS 2025, arXiv:2505.06708)
- Official code: [`qiuzh20/gated_attention`](https://github.com/qiuzh20/gated_attention)
  (tracked here as a git **submodule** under `src/official`)

## What this repo contains

| Path | Purpose |
| --- | --- |
| `src/official/` | **Submodule** — the authors' reference implementation (Apache-2.0). |
| `src/our/model_builder.py` | Builds `baseline` / `headwise` / `elementwise` variants from the official classes. |
| `src/our/validate_gate.py` | CPU sanity check that the gate logic is correct (more params, extended `q_proj`, gate changes output). |
| `src/our/eval_attention.py` | Phase A: dataset PPL, attention sink, gates, activations, maps. |
| `src/our/prep_data.py`, `train.py` | Streaming data prep + token-matched Phase B training. |
| `src/our/analyze_checkpoints.py` | Checkpoint dynamics and 512–4096 context sweeps. |
| `src/our/metrics.py`, `compare_runs.py` | Shared metrics and final result aggregation. |
| `scripts/run_smoke.sh`, `run_eval_1b.sh`, `run_train.sh` | One-click launchers. |
| `docs/reproduction_plan.md` | Complete experiment design, metrics, commands, and limits. |
| `requirements.txt` | Pinned dependencies. |

## Reproduction roadmap

- **Phase A — validate the official 1B models** (`scripts/run_eval_1b.sh`):
  downloads the released `1B_baseline` / `1B_gate_headwise` /
  `1B_gate_elementwise`, measures the first-token attention rate (attention
  sink) and PPL, and plots per-layer attention maps.
- **Phase B — self-train a small model** (`scripts/run_train.sh`): trains the
  three variants from scratch on FineWeb-Edu with the same token budget, then
  replays model-only snapshots to track sink, gates, activations, and context
  extrapolation over training.
- **Phase C — long-context extrapolation** (RULER): planned.

## ⚠️ Environment pin (important)

The official `modeling_qwen3.py` targets **transformers 4.x**. transformers ≥ 5.0
silently breaks it (`config.qkv_bias`, `ROPE_INIT_FUNCTIONS['default']`, implicit
`pad_token_id` all gone). We verified the gate sanity check only passes on
**transformers==4.51.3**, so the dependency is pinned in `requirements.txt`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # on the 4xL20 (CUDA 12)
pip install -r requirements.txt
```

## Quick start

```bash
# CPU smoke test (gate correctness) — runs anywhere torch is installed
python src/our/validate_gate.py

# End-to-end TinyStories smoke test
bash scripts/run_smoke.sh

# On the GPU machine
bash scripts/run_eval_1b.sh
LAUNCH="accelerate launch --num_processes 4" bash scripts/run_train.sh
```

Read [`docs/reproduction_plan.md`](docs/reproduction_plan.md) before starting a
paid or long-running experiment.

## Safe upstream entry points

`src/official` is a pinned third-party submodule. Do not execute its `demo.py`
or notebooks directly: the demo enables remote code and the curve notebook uses
pickle-backed `torch.load`. Use the local implementations instead:

```bash
python src/our/eval_attention.py --help
python src/our/plot_official_figures.py
```

All training and evaluation commands in this repository import the compatibility
layer in `src/our/model_builder.py`, which repairs the pinned upstream config and
Flash Attention edge cases without carrying an unpublished submodule commit.

## License

- `src/official/` is the authors' code under **Apache-2.0** (see its `LICENSE`).
- Everything under `src/our/`, `scripts/`, `docs/`, and this README is released
  under **MIT**.
