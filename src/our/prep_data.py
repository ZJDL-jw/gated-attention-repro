"""Prepare a small, clean tokenized corpus for Phase B self-training.

We tokenize a public text dataset with the Qwen3 tokenizer, then pack the
token ids into fixed-length blocks (the standard GPT-style "document
packing" used for causal LM pretraining). The result is saved with
`datasets.Dataset.save_to_disk`, which `train.py` loads directly.

Run on the 4xL20 (or any machine with internet + the CUDA env):
    python src/our/prep_data.py \
        --dataset_name roneneldan/TinyStories \
        --tokenizer Qwen/Qwen3-0.6B \
        --block_size 2048 \
        --max_tokens 50000000 \
        --output_dir data/tinystories_50M

Why TinyStories? It is small (a few GB), clean English, and more than enough
to *show the trend* (gated attention trains more stably / sinks less). For a
stronger result you can swap --dataset_name for e.g.
HuggingFaceFW/fineweb-edu (and raise --max_tokens).
"""
import argparse
import os

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="roneneldan/TinyStories")
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--max_tokens", type=int, default=50_000_000,
                    help="stop after accumulating ~this many tokens")
    ap.add_argument("--streaming", action="store_true",
                    help="use streaming mode for very large datasets")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[prep] loading {args.dataset_name} ({args.dataset_split}) ...")
    ds = load_dataset(args.dataset_name, split=args.dataset_split,
                      streaming=args.streaming)
    if args.streaming:
        ds = ds.take(int(args.max_tokens * 1.5))  # over-fetch, we cap below

    ids: list[int] = []
    for ex in ds:
        text = ex.get(args.text_field)
        if not text:
            continue
        ids.extend(tok(text, return_attention_mask=False)["input_ids"])
        if len(ids) >= args.max_tokens:
            break

    ids = ids[: args.max_tokens]
    print(f"[prep] collected {len(ids):,} raw tokens")

    # Pack into contiguous blocks of block_size (drop the final short tail).
    n_blocks = len(ids) // args.block_size
    blocks = [
        ids[i * args.block_size: (i + 1) * args.block_size]
        for i in range(n_blocks)
    ]
    print(f"[prep] {n_blocks:,} blocks of length {args.block_size}")

    block_ds = Dataset.from_dict({"input_ids": blocks})
    os.makedirs(args.output_dir, exist_ok=True)
    block_ds.save_to_disk(args.output_dir)

    meta = dict(
        dataset_name=args.dataset_name,
        tokenizer=args.tokenizer,
        block_size=args.block_size,
        n_blocks=n_blocks,
        total_tokens=n_blocks * args.block_size,
    )
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"[prep] saved -> {args.output_dir}")
    print(f"[prep] meta: {meta}")


if __name__ == "__main__":
    main()
