"""Compare the three Phase B training runs: loss curves, attention-sink
trajectories, and final sink rates.

Run after run_train.sh:
    python src/our/compare_runs.py --out_root outputs --out results/phaseB

Produces:
    results/phaseB/loss_curves.png         : training loss vs step (3 variants)
    results/phaseB/sink_curves.png         : attention-sink vs step (3 variants)
    results/phaseB/summary.md              : final mean first-token rates table
"""
import argparse
import glob
import json
import os

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
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="outputs/ from run_train.sh")
    ap.add_argument("--out", default="results/phaseB")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    variants = ["baseline", "headwise", "elementwise"]
    curves, finals = {}, {}

    for v in variants:
        curve_path = os.path.join(args.out_root, v, f"sink_curve_{v}.jsonl")
        if not os.path.exists(curve_path):
            print(f"[compare] skipping {v}: no {curve_path}")
            continue
        rows = read_jsonl(curve_path)
        steps = [r["step"] for r in rows]
        loss = [r.get("loss") for r in rows]
        sink = [r["mean_first_token_rate"] for r in rows]
        curves[v] = (steps, loss, sink)

        res_path = os.path.join(args.out_root, v, f"results_{v}_trained.json")
        if os.path.exists(res_path):
            with open(res_path) as f:
                finals[v] = json.load(f)["mean_first_token_rate"]

    if not curves:
        print("[compare] nothing to compare yet.")
        return

    # loss curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v, (steps, loss, _sink) in curves.items():
        if any(l is not None for l in loss):
            ax.plot(steps, loss, label=v, color=VARIANT_COLORS[v])
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("Phase B: training loss")
    ax.legend()
    plt.tight_layout()
    loss_png = os.path.join(args.out, "loss_curves.png")
    plt.savefig(loss_png, dpi=120)
    plt.close()

    # sink curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v, (_steps, _loss, sink) in curves.items():
        ax.plot(_steps, sink, label=v, color=VARIANT_COLORS[v])
    ax.set_xlabel("step")
    ax.set_ylabel("mean first-token attention rate")
    ax.set_title("Phase B: attention sink during training")
    ax.legend()
    plt.tight_layout()
    sink_png = os.path.join(args.out, "sink_curves.png")
    plt.savefig(sink_png, dpi=120)
    plt.close()

    # summary table
    lines = ["# Phase B comparison (self-trained, ~200M params)\n",
             "| variant | final mean first-token rate |",
             "| --- | --- |"]
    for v in variants:
        if v in finals:
            lines.append(f"| {v} | {finals[v]:.4f} |")
    lines.append("")
    lines.append(f"![loss](loss_curves.png)\n")
    lines.append(f"![sink](sink_curves.png)\n")
    summary = os.path.join(args.out, "summary.md")
    with open(summary, "w") as f:
        f.write("\n".join(lines))

    print(f"[compare] loss  -> {loss_png}")
    print(f"[compare] sink  -> {sink_png}")
    print(f"[compare] table -> {summary}")
    print("\nFinal attention-sink rates:")
    for v in variants:
        if v in finals:
            print(f"  {v:12s} {finals[v]:.4f}")


if __name__ == "__main__":
    main()
