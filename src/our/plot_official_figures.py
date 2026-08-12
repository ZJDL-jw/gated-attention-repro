"""Safely regenerate the figures shipped as executable upstream notebooks."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


BENCHMARKS = {
    "baseline": {"MMLU": 58.79, "avg PPL": 6.026},
    "SDPA": {"MMLU": 60.82, "avg PPL": 5.761},
    "value": {"MMLU": 59.17, "avg PPL": 5.820},
    "key": {"MMLU": 59.18, "avg PPL": 6.016},
    "query": {"MMLU": 58.74, "avg PPL": 5.981},
    "dense": {"MMLU": 59.41, "avg PPL": 6.017},
}


def smooth(values, alpha=0.9):
    values = np.asarray(values)
    if values.size == 0:
        return values
    result = np.zeros_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * result[index - 1] + (1 - alpha) * values[index]
    return result


def load_curve(path: Path):
    """Load tensor-only curve data without permitting pickle code execution."""
    return torch.load(path, map_location="cpu", weights_only=True)


def plot_loss_curves(figs_dir: Path, output_dir: Path) -> Path:
    baseline = load_curve(figs_dir / "tb_curves" / "3T-baseline.pt")
    gate = load_curve(figs_dir / "tb_curves" / "3T-gate.pt")
    fig, axis = plt.subplots(figsize=(7, 9))
    axis.plot(
        baseline["steps"],
        smooth(baseline["lm_losses"]),
        label="Baseline",
        alpha=0.3,
        color="black",
    )
    axis.plot(
        gate["steps"],
        smooth(gate["lm_losses"]),
        label="SDPA output gate G1",
        alpha=0.5,
        color="#4C72B0",
    )
    axis.set(xlabel="Steps (total 360k)", ylabel="LM loss", ylim=(1.965, 2.3))
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axis.legend()
    fig.tight_layout()
    destination = output_dir / "lm_loss.pdf"
    fig.savefig(destination, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_benchmark_comparison(output_dir: Path) -> Path:
    baseline = BENCHMARKS["baseline"]
    names = ["SDPA", "value", "key", "query", "dense"]
    colors = ["#4C72B0", "#55A868", "#8DA0CB", "#E78AC3", "#CC613D"]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    for axis, metric in zip(axes, ("avg PPL", "MMLU")):
        base_value = baseline[metric]
        values = [BENCHMARKS[name][metric] for name in names]
        axis.axhline(base_value, color="black", linestyle="--", label="baseline")
        axis.bar(
            positions,
            [value - base_value for value in values],
            bottom=base_value,
            color=colors,
            alpha=0.5,
        )
        axis.set_ylabel(metric)
        axis.set_xticks(positions, labels=names)
    axes[0].legend()
    fig.tight_layout()
    destination = output_dir / "benchmark_ppl_comparison.pdf"
    fig.savefig(destination, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs_dir", default="src/official/figs")
    parser.add_argument("--output_dir", default="results/official_figures")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(Path(args.figs_dir), output_dir)
    plot_benchmark_comparison(output_dir)
    print(f"[figures] saved -> {output_dir}")


if __name__ == "__main__":
    main()
