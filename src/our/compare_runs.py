"""Aggregate training logs and dynamics probes across Phase B runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT_COLORS = {
    "baseline": "#d62728",
    "headwise": "#2ca02c",
    "elementwise": "#1f77b4",
}


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_runs(root: Path):
    runs = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        variant = manifest["variant"]
        dynamics_path = run_dir / "analysis" / f"dynamics_{variant}.jsonl"
        state_path = run_dir / "trainer_state.json"
        runs.append(
            {
                "dir": run_dir,
                "manifest": manifest,
                "variant": variant,
                "label": f"{variant}/seed-{manifest.get('seed', '?')}",
                "dynamics": read_jsonl(dynamics_path) if dynamics_path.exists() else [],
                "state": json.loads(state_path.read_text()) if state_path.exists() else {},
            }
        )
    return runs


def count_loss_spikes(log_history, window=20):
    """Count large positive deviations from a trailing robust loss baseline."""
    losses = [row["loss"] for row in log_history if "loss" in row]
    spikes = 0
    for index in range(window, len(losses)):
        history = losses[index - window : index]
        median = statistics.median(history)
        mad = statistics.median(abs(value - median) for value in history)
        margin = max(6 * mad, 0.1 * median)
        spikes += losses[index] > median + margin
    return int(spikes)


def plot_training(runs, output_dir):
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for run in runs:
        tokens_per_step = run["manifest"].get("tokens_per_step", 0)
        logs = [row for row in run["state"].get("log_history", []) if "loss" in row]
        if not logs:
            continue
        axis.plot(
            [row["step"] * tokens_per_step for row in logs],
            [row["loss"] for row in logs],
            label=run["label"],
            color=VARIANT_COLORS.get(run["variant"]),
            alpha=0.85,
        )
        plotted = True
    axis.set(xlabel="processed tokens", ylabel="training loss", title="Training loss")
    if plotted:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "train_loss.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for run in runs:
        tokens_per_step = run["manifest"].get("tokens_per_step", 0)
        logs = [
            row for row in run["state"].get("log_history", []) if "eval_loss" in row
        ]
        if not logs:
            continue
        axis.plot(
            [row["step"] * tokens_per_step for row in logs],
            [math.exp(row["eval_loss"]) for row in logs],
            marker="o",
            label=run["label"],
            color=VARIANT_COLORS.get(run["variant"]),
        )
        plotted = True
    axis.set(
        xlabel="processed tokens", ylabel="validation PPL", title="Validation PPL"
    )
    if plotted:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_ppl.png", dpi=140)
    plt.close(fig)


def plot_dynamics(runs, output_dir):
    specifications = [
        (
            "sink_over_tokens.png",
            "Attention sink over training",
            "prefix-excluded first-token attention",
            lambda row: row["probe"]["prefix_excluded_mean"],
        ),
        (
            "gate_sparsity_over_tokens.png",
            "Gate sparsity over training",
            "fraction of gates < 0.1",
            lambda row: row["probe"]["gate"]["overall"]["fraction_lt_0_1"],
        ),
        (
            "activation_over_tokens.png",
            "Residual activation over training",
            "max |FFN residual| across layers",
            lambda row: max(
                stats["max_abs"]
                for stats in row["probe"]["activations"]["ffn_residual"].values()
            ),
        ),
    ]
    for filename, title, ylabel, extractor in specifications:
        fig, axis = plt.subplots(figsize=(8, 5))
        plotted = False
        for run in runs:
            points = []
            for row in run["dynamics"]:
                value = extractor(row)
                if value is not None:
                    points.append((row["processed_tokens"], value))
            if not points:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=run["label"],
                color=VARIANT_COLORS.get(run["variant"]),
            )
            plotted = True
        axis.set(xlabel="processed tokens", ylabel=ylabel, title=title)
        if plotted:
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)


def plot_context_results(runs, output_dir):
    fig, axis = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for run in runs:
        for length in (512, 1024, 2048, 4096):
            points = []
            for row in run["dynamics"]:
                metric = row["ppl_by_context"].get(str(length))
                if metric:
                    points.append((row["processed_tokens"], metric["ppl"]))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    label=f"{run['label']} / {length}",
                    color=VARIANT_COLORS.get(run["variant"]),
                    linestyle={512: ":", 1024: "-.", 2048: "-", 4096: "--"}[length],
                )
                plotted = True
    axis.set(
        xlabel="processed tokens",
        ylabel="validation PPL",
        title="Context-length PPL during training",
        yscale="log",
    )
    if plotted:
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "context_ppl_over_tokens.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for run in runs:
        if not run["dynamics"]:
            continue
        final = run["dynamics"][-1]["ppl_by_context"]
        lengths = sorted(int(length) for length in final)
        axis.plot(
            lengths,
            [final[str(length)]["ppl"] for length in lengths],
            marker="o",
            label=run["label"],
            color=VARIANT_COLORS.get(run["variant"]),
        )
        plotted = True
    axis.set(
        xlabel="context length",
        ylabel="final validation PPL",
        title="Final length degradation",
        xscale="log",
    )
    axis.set_xticks([512, 1024, 2048, 4096], labels=["512", "1024", "2048", "4096"])
    axis.minorticks_off()
    if plotted:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "length_degradation_final.png", dpi=140)
    plt.close(fig)


def write_summary(runs, output_dir):
    rows = []
    for run in runs:
        if not run["dynamics"]:
            continue
        final = run["dynamics"][-1]
        gate = final["probe"]["gate"]["overall"]
        row = {
            "variant": run["variant"],
            "seed": run["manifest"].get("seed"),
            "parameters": run["manifest"].get("parameter_count"),
            "processed_tokens": final["processed_tokens"],
            "sink_paper_style": final["probe"]["paper_style_mean"],
            "sink_prefix_excluded": final["probe"]["prefix_excluded_mean"],
            "gate_mean": gate["mean"],
            "gate_fraction_lt_0_1": gate["fraction_lt_0_1"],
            "loss_spikes": count_loss_spikes(
                run["state"].get("log_history", [])
            ),
        }
        for length, metric in final["ppl_by_context"].items():
            row[f"ppl_{length}"] = metric["ppl"]
        rows.append(row)

    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Phase B summary", "", "| variant | seed | tokens | sink | PPL@2048 | PPL@4096 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['processed_tokens']} | "
            f"{row['sink_prefix_excluded']:.4f} | {row.get('ppl_2048', float('nan')):.3f} | "
            f"{row.get('ppl_4096', float('nan')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate across seeds",
            "",
            "| variant | n | sink (mean +/- SD) | PPL@2048 (mean +/- SD) | PPL@4096 (mean +/- SD) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]

        def mean_sd(key):
            values = [float(row[key]) for row in group if key in row]
            if not values:
                return float("nan"), float("nan")
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            return mean, sd

        sink_mean, sink_sd = mean_sd("sink_prefix_excluded")
        ppl_2048_mean, ppl_2048_sd = mean_sd("ppl_2048")
        ppl_4096_mean, ppl_4096_sd = mean_sd("ppl_4096")
        metric_text = lambda value, digits: f"{value:.{digits}f}" if math.isfinite(value) else "n/a"
        lines.append(
            f"| {variant} | {len(group)} | {metric_text(sink_mean, 4)} +/- {metric_text(sink_sd, 4)} | "
            f"{metric_text(ppl_2048_mean, 3)} +/- {metric_text(ppl_2048_sd, 4)} | "
            f"{metric_text(ppl_4096_mean, 3)} +/- {metric_text(ppl_4096_sd, 4)} |"
        )
    lines.extend(
        [
            "",
            "These runs use about 200M parameters and 500M training tokens. "
            "Baseline and elementwise use three seeds; headwise uses one seed. "
            "The 4096-token evaluation is zero-shot RoPE extrapolation beyond the "
            "2048-token training length, not a substitute for long-context continued "
            "pretraining or RULER. Treat the curves as dynamics and correlation evidence, "
            "not a causal proof or a full-scale reproduction.",
            "",
            "![training loss](train_loss.png)",
            "",
            "![validation PPL](validation_ppl.png)",
            "",
            "![sink](sink_over_tokens.png)",
            "",
            "![context PPL](context_ppl_over_tokens.png)",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--out", default="results/phaseB")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(Path(args.out_root))
    if not runs:
        raise FileNotFoundError(f"No run_manifest.json files under {args.out_root}")
    plot_training(runs, output_dir)
    plot_dynamics(runs, output_dir)
    plot_context_results(runs, output_dir)
    write_summary(runs, output_dir)
    print(f"[compare] {len(runs)} runs -> {output_dir}")


if __name__ == "__main__":
    main()
