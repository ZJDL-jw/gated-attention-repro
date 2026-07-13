"""Aggregate per-variant Phase A results into a comparison table + bar chart.

Run:
    python src/our/aggregate_results.py --results_dir results/phaseA
"""
import argparse
import json
import os
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, "results_*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))

    if not rows:
        print("No results_*.json found in", args.results_dir)
        return

    print("\n=== Attention-sink comparison (mean first-token attention rate) ===")
    print(f"{'variant':16s} {'mean_rate':>10s}   {'PPL':>10s}")
    print("-" * 42)
    for r in rows:
        ppl = r.get("ppl")
        print(f"{r['variant']:16s} {r['mean_first_token_rate']:10.4f}   {('%.3f' % ppl) if ppl else '  n/a':>10s}")

    # bar chart
    variants = [r["variant"] for r in rows]
    rates = [r["mean_first_token_rate"] for r in rows]
    colors = ["#d62728" if v == "baseline" else "#2ca02c" for v in variants]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(variants, rates, color=colors)
    ax.set_ylabel("mean first-token attention rate")
    ax.set_title("Attention sink: baseline vs gated")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.005, f"{r:.3f}", ha="center")
    ax.set_ylim(0, max(rates) * 1.2 + 0.01)
    plt.tight_layout()
    out = os.path.join(args.results_dir, "attention_sink_comparison.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nbar chart -> {out}")


if __name__ == "__main__":
    main()
