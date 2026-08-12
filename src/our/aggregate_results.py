"""Aggregate Phase A result JSON files into a table and comparison chart."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def flatten_result(result):
    dataset = result.get("dataset")
    if dataset:
        probe = dataset["probe"]
        gate = probe["gate"]["overall"]
        return {
            "variant": result["variant"],
            "parameters": result.get("parameter_count"),
            "ppl": dataset["ppl"]["ppl"],
            "sink_paper_style": probe["paper_style_mean"],
            "sink_prefix_excluded": probe["prefix_excluded_mean"],
            "gate_mean": gate["mean"],
            "gate_fraction_lt_0_1": gate["fraction_lt_0_1"],
        }
    return {
        "variant": result["variant"],
        "parameters": result.get("parameter_count"),
        "ppl": result.get("ppl", result.get("legacy_text_ppl")),
        "sink_paper_style": result.get(
            "mean_first_token_rate", result.get("prompt", {}).get("paper_style_mean")
        ),
        "sink_prefix_excluded": None,
        "gate_mean": None,
        "gate_fraction_lt_0_1": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    rows = [
        flatten_result(json.loads(path.read_text()))
        for path in sorted(results_dir.glob("results_*.json"))
    ]
    if not rows:
        raise FileNotFoundError(f"No results_*.json under {results_dir}")

    columns = list(rows[0])
    with (results_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Phase A: official checkpoint comparison ===")
    print(f"{'variant':18s} {'sink':>10s} {'trimmed':>10s} {'PPL':>10s}")
    for row in rows:
        trimmed = row["sink_prefix_excluded"]
        ppl = row["ppl"]
        print(
            f"{row['variant']:18s} {row['sink_paper_style']:10.4f} "
            f"{trimmed if trimmed is not None else float('nan'):10.4f} "
            f"{ppl if ppl is not None else float('nan'):10.3f}"
        )

    variants = [row["variant"] for row in rows]
    paper = [row["sink_paper_style"] for row in rows]
    trimmed = [row["sink_prefix_excluded"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar([x - width / 2 for x in positions], paper, width, label="paper-style")
    if all(value is not None for value in trimmed):
        axis.bar(
            [x + width / 2 for x in positions],
            trimmed,
            width,
            label="exclude first 4 queries",
        )
    axis.set_xticks(positions, labels=variants)
    axis.set_ylabel("mean first-token attention")
    axis.set_title("Phase A attention sink")
    axis.legend()
    fig.tight_layout()
    chart = results_dir / "attention_sink_comparison.png"
    fig.savefig(chart, dpi=140)
    plt.close(fig)
    print(f"summary -> {results_dir / 'summary.csv'}")
    print(f"chart   -> {chart}")


if __name__ == "__main__":
    main()
