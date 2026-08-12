"""Prepare deterministic, packed train/validation data for causal-LM training.

The implementation is intentionally streaming-friendly: it never materializes
the requested token budget as a Python list. Documents are separated with EOS,
packed into fixed-size blocks, written incrementally to Arrow, then exposed as
a ``DatasetDict`` with ``train`` and ``validation`` splits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer

from data_utils import pack_token_sequences


def safe_output_path(path: str | Path) -> Path:
    """Resolve an output path while rejecting destructive overwrite targets."""
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError(f"Refusing to use symlink as output directory: {raw_path}")

    resolved = raw_path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    protected = {
        Path(resolved.anchor),
        Path.home().resolve(),
        Path.cwd().resolve(),
        project_root,
        *project_root.parents,
    }
    if resolved in protected:
        raise ValueError(f"Refusing destructive output directory: {resolved}")
    return resolved


def commit_dataset(dataset, staged_dir: Path, output_dir: Path, meta: dict) -> None:
    """Write a complete dataset in staging, then atomically publish it."""
    dataset.save_to_disk(str(staged_dir))
    with (staged_dir / "meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    backup_dir = staged_dir.parent / "previous"
    if output_dir.exists():
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staged_dir, output_dir)
    except BaseException:
        if backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _load_source(dataset_name, dataset_config, dataset_split, streaming):
    args = [dataset_name]
    if dataset_config:
        args.append(dataset_config)
    return load_dataset(*args, split=dataset_split, streaming=streaming)


def iter_packed_blocks(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    tokenizer_name: str,
    text_field: str,
    block_size: int,
    total_blocks: int,
    streaming: bool,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[dict[str, list[int]]]:
    """Yield at most ``total_blocks`` token blocks without corpus-sized RAM."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name!r} has no eos_token_id")

    source = _load_source(dataset_name, dataset_config, dataset_split, streaming)
    if shuffle_buffer > 0:
        if streaming:
            source = source.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            source = source.shuffle(seed=seed)

    def token_sequences():
        for example in source:
            text = example.get(text_field)
            if not text:
                continue
            yield tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

    yield from pack_token_sequences(
        token_sequences(), tokenizer.eos_token_id, block_size, total_blocks
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--train_tokens", type=int, default=20_000_000)
    parser.add_argument("--validation_tokens", type=int, default=2_000_000)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream the source dataset (recommended for FineWeb-Edu)",
    )
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--shuffle_buffer", type=int, default=10_000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_size < 2:
        raise ValueError("--block_size must be at least 2")
    if args.train_tokens < args.block_size:
        raise ValueError("--train_tokens must contain at least one full block")
    if args.validation_tokens < args.block_size:
        raise ValueError("--validation_tokens must contain at least one full block")

    train_blocks = args.train_tokens // args.block_size
    validation_blocks = args.validation_tokens // args.block_size
    total_blocks = train_blocks + validation_blocks

    output_dir = safe_output_path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace it"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Validate the tokenizer early so a missing EOS fails before corpus work.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {args.tokenizer!r} has no eos_token_id")

    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    cache_dir = build_dir / "cache"
    staged_dir = build_dir / "dataset"
    features = Features(
        {"input_ids": Sequence(Value("int32"), length=args.block_size)}
    )
    print(
        f"[prep] dataset={args.dataset_name} split={args.dataset_split} "
        f"blocks={total_blocks:,} block_size={args.block_size}"
    )

    try:
        packed = Dataset.from_generator(
            iter_packed_blocks,
            gen_kwargs=dict(
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                tokenizer_name=args.tokenizer,
                text_field=args.text_field,
                block_size=args.block_size,
                total_blocks=total_blocks,
                streaming=args.streaming,
                seed=args.seed,
                shuffle_buffer=args.shuffle_buffer,
            ),
            features=features,
            cache_dir=str(cache_dir),
            keep_in_memory=False,
        )
        if len(packed) < total_blocks:
            raise RuntimeError(
                f"Source ended after {len(packed):,} blocks; requested {total_blocks:,}"
            )

        # The source is shuffled first, so reserving the leading blocks gives a
        # deterministic, disjoint validation stream without another corpus pass.
        validation = packed.select(range(validation_blocks))
        train = packed.select(range(validation_blocks, total_blocks))
        meta = dict(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            tokenizer=args.tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            block_size=args.block_size,
            seed=args.seed,
            streaming=args.streaming,
            shuffle_buffer=args.shuffle_buffer,
            train_blocks=len(train),
            validation_blocks=len(validation),
            train_tokens=len(train) * args.block_size,
            validation_tokens=len(validation) * args.block_size,
        )
        dataset = DatasetDict(train=train, validation=validation)
        commit_dataset(dataset, staged_dir, output_dir, meta)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    print(f"[prep] saved -> {output_dir}")
    print(f"[prep] train={meta['train_tokens']:,} validation={meta['validation_tokens']:,}")


if __name__ == "__main__":
    main()
