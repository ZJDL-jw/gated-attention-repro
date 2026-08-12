"""Replay model-only training snapshots and measure attention dynamics."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk

from eval_attention import load_model
from metrics import evaluate_attention_and_probes, evaluate_context_lengths


def _step_from_path(path: Path) -> int:
    return int(path.name.split("-", 1)[1])


def analyze_run(
    run_dir: str | Path,
    data_dir: str | Path,
    variant: str,
    context_lengths: list[int],
    ppl_tokens: int,
    sink_length: int,
    sink_samples: int,
    device: str,
) -> Path:
    run_dir = Path(run_dir)
    data = load_from_disk(str(data_dir))
    if not isinstance(data, DatasetDict) or "validation" not in data:
        raise TypeError("Checkpoint analysis requires a validation Dataset split")
    validation = data["validation"]

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    tokens_per_step = int(manifest.get("tokens_per_step", 0))

    snapshots = sorted(
        (run_dir / "analysis_snapshots").glob("step-*"), key=_step_from_path
    )
    if not snapshots:
        raise FileNotFoundError(f"No analysis snapshots under {run_dir}")

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / f"dynamics_{variant}.jsonl"
    rows = []

    for snapshot_index, snapshot in enumerate(snapshots):
        step = _step_from_path(snapshot)
        print(f"[analyze:{variant}] step={step} model={snapshot}")
        model = load_model(str(snapshot), device)
        ppl_by_context = evaluate_context_lengths(
            model,
            validation,
            context_lengths=context_lengths,
            device=device,
            max_tokens_per_length=ppl_tokens,
            batch_size=1,
        )
        probe = evaluate_attention_and_probes(
            model,
            validation,
            context_length=sink_length,
            device=device,
            max_samples=sink_samples,
            exclude_query_prefix=4,
        )

        row = {
            "variant": variant,
            "step": step,
            "processed_tokens": step * tokens_per_step if tokens_per_step else None,
            "ppl_by_context": {
                str(item["context_length"]): item for item in ppl_by_context
            },
            "probe": probe,
            "sink_by_context_final": None,
        }

        # Full attention is quadratic. Measure it by length only for the final
        # snapshot, with one sample at long lengths, while every snapshot still
        # receives the complete PPL sweep and fixed-length dynamics probe.
        if snapshot_index == len(snapshots) - 1:
            sink_by_context = {}
            for length in context_lengths:
                samples = sink_samples if length <= sink_length else 1
                length_probe = evaluate_attention_and_probes(
                    model,
                    validation,
                    context_length=length,
                    device=device,
                    max_samples=samples,
                    exclude_query_prefix=4,
                )
                sink_by_context[str(length)] = {
                    "paper_style_mean": length_probe["paper_style_mean"],
                    "prefix_excluded_mean": length_probe["prefix_excluded_mean"],
                    "samples": length_probe["samples"],
                }
            row["sink_by_context_final"] = sink_by_context

        rows.append(row)
        with output_path.open("w") as handle:
            for completed in rows:
                handle.write(json.dumps(completed) + "\n")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[analyze:{variant}] results -> {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--context_lengths", default="512,1024,2048,4096")
    parser.add_argument("--ppl_tokens", type=int, default=131_072)
    parser.add_argument("--sink_length", type=int, default=512)
    parser.add_argument("--sink_samples", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_run(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        variant=args.variant,
        context_lengths=[int(value) for value in args.context_lengths.split(",")],
        ppl_tokens=args.ppl_tokens,
        sink_length=args.sink_length,
        sink_samples=args.sink_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
