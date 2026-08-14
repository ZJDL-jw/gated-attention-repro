# Reproduction environment manifest

Captured on 2026-08-14 at 14:24:56 +08:00 from the GPU worker used for the
published smoke, Phase A, and Phase B artifacts. This is a curated manifest: it
does not contain credentials, complete environment variables, caches, model
weights, datasets, or machine-specific user directories.

## Host and accelerator

| Item | Captured value |
| --- | --- |
| Operating system | Ubuntu 22.04.5 LTS |
| Kernel | Linux 6.8.0-65-generic, x86_64 |
| Physical GPUs | 4 × NVIDIA L20, 46,068 MiB each |
| NVIDIA driver | 560.35.03 |
| PyTorch CUDA runtime | 12.1 |
| CUDA available to PyTorch | yes; 4 devices visible at capture time |
| Python | 3.10.12, GCC 11.4.0 |

The worker physically exposes four L20 GPUs, but the stored `run_manifest.json`
files record `world_size=2` and two L20 devices for every published smoke and
formal training run. Results must therefore be interpreted as 2-GPU runs, not
4-GPU runs.

## Python packages

| Package | Version |
| --- | --- |
| torch | 2.5.1+cu121 |
| transformers | 4.51.3 |
| datasets | 3.6.0 |
| accelerate | 1.14.0 |
| numpy | 1.26.4 |
| matplotlib | 3.8.3 |
| safetensors | 0.8.0 |
| huggingface-hub | 0.36.2 |

These values match the run manifests and `requirements.txt`. The virtual
environment was created without an importable `pip` module at capture time;
package versions were read through Python package metadata instead.

## Code identity

| Component | Revision |
| --- | --- |
| Public base used by the experiment checkout | `0e3dec674f2d938e4e369f8c2a656391ee8e67e2` |
| Published result/code snapshot | `27eed046c8970cac9226abd4dbc45685b46f6572` |
| `src/official` submodule | `f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2` |

The result snapshot publishes the analysis/aggregation change that was present
in the experiment workspace. Unpublished raw logs are not part of the code
identity above.

## Data and training topology

| Item | Value |
| --- | --- |
| Formal dataset | `HuggingFaceFW/fineweb-edu`, config `default` |
| Tokenizer | `Qwen/Qwen3-0.6B` |
| Block size | 2,048 |
| Data shuffle seed | 20 |
| Formal train tokens on disk | 499,998,720 |
| Formal validation tokens on disk | 7,999,488 |
| Formal train/validation blocks | 244,140 / 3,906 |
| Actual distributed world size | 2 |
| Tokens per formal optimizer step | 16,384 |
| Planned processed tokens per formal run | 500,006,912 |
| Formal optimizer steps | 30,518 |
| Snapshots per formal run | 9, including step 0 and the final step |

The Hugging Face cache was explicitly shared at
`/mnt/ARD340/.hf_cache`; no repository-local `.huggingface` cache was required
or published.

## Published run coverage

- TinyStories smoke: baseline, headwise, and elementwise at seed 20.
- Official 1B Phase A: baseline, headwise gate, and elementwise gate.
- FineWeb-Edu Phase B: baseline seeds 20/21/22, elementwise seeds 20/21/22,
  and headwise seed 20.
- All seven formal runs reached step 30,518 and produced 9/9 planned analysis
  snapshots. Exact final training statistics are in
  `logs/reproduction_summary.log`; evaluation metrics are in `results/`.
