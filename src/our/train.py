"""Train one gated-attention variant with reproducible, token-based settings."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from metrics import read_data_meta
from model_builder import VARIANTS, build_model


GATE_FLAGS = ("headwise_attn_output_gate", "elementwise_attn_output_gate")


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    for key in GATE_FLAGS:
        config.pop(key, None)
    # Human-readable JSON notes are not model constructor arguments.
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def validate_output_dir(path, resume_from_checkpoint=None):
    """Reject accidental from-scratch reuse of a non-empty run directory."""
    output_dir = Path(path)
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and resume_from_checkpoint is None
    ):
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new --output_dir or pass "
            "--resume_from_checkpoint"
        )
    return output_dir


def validate_tokenizer(data_meta, tokenizer_name, tokenizer, config):
    """Ensure stored token ids and the tokenizer saved with the run agree."""
    prepared_with = data_meta.get("tokenizer")
    if prepared_with and prepared_with != tokenizer_name:
        raise ValueError(
            f"Dataset was prepared with tokenizer {prepared_with!r}, but training "
            f"requested {tokenizer_name!r}"
        )
    prepared_eos = data_meta.get("eos_token_id")
    if prepared_eos is not None and tokenizer.eos_token_id != prepared_eos:
        raise ValueError(
            f"Dataset EOS id {prepared_eos} does not match tokenizer EOS id "
            f"{tokenizer.eos_token_id}"
        )
    if len(tokenizer) > int(config.vocab_size):
        raise ValueError(
            f"Tokenizer size {len(tokenizer)} exceeds model vocab_size "
            f"{config.vocab_size}"
        )


def mixed_precision_flags(bf16_requested, use_cuda):
    """Select BF16 when supported, otherwise fall back to FP16 on CUDA."""
    bf16_supported = use_cuda and torch.cuda.is_bf16_supported()
    return {
        "bf16": bool(bf16_requested and bf16_supported),
        "fp16": bool(bf16_requested and use_cuda and not bf16_supported),
    }


class AnalysisSnapshotCallback(TrainerCallback):
    """Save model-only snapshots at the same steps as Trainer checkpoints."""

    def __init__(self, output_dir, milestone_steps):
        self.root = Path(output_dir) / "analysis_snapshots"
        self.milestones = set(milestone_steps)
        self.saved = set()

    def _save(self, state, model):
        step = state.global_step
        if step in self.saved or step not in self.milestones:
            return
        _barrier()
        if state.is_world_process_zero:
            destination = self.root / f"step-{step:08d}"
            destination.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(destination, safe_serialization=True)
            (destination / "snapshot.json").write_text(
                json.dumps({"step": step}, indent=2)
            )
            print(f"[snapshot] step={step} -> {destination}")
        _barrier()
        self.saved.add(step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._save(state, model)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._save(state, model)
        return control


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--target_train_tokens", type=int, default=20_000_000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--analysis_checkpoints", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--trainer_eval_tokens", type=int, default=1_048_576)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--analyze_after_train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--analysis_context_lengths", default="512,1024,2048,4096")
    parser.add_argument("--analysis_ppl_tokens", type=int, default=131_072)
    parser.add_argument("--analysis_sink_length", type=int, default=512)
    parser.add_argument("--analysis_sink_samples", type=int, default=8)
    return parser.parse_args()


def package_versions():
    versions = {}
    for package in ("torch", "transformers", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main():
    args = parse_args()
    set_seed(args.seed)  # Must happen before model construction.
    output_dir = Path(args.output_dir)
    if int(os.environ.get("RANK", "0")) == 0:
        output_dir = validate_output_dir(
            args.output_dir, resume_from_checkpoint=args.resume_from_checkpoint
        )

    data = load_from_disk(args.data_dir)
    if not isinstance(data, DatasetDict):
        raise TypeError(
            "Training data must be a DatasetDict from the new prep_data.py"
        )
    if "train" not in data or "validation" not in data:
        raise KeyError("Training data requires train and validation splits")

    data_meta = read_data_meta(args.data_dir)
    block_size = int(data_meta.get("block_size", len(data["train"][0]["input_ids"])))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = (
        world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
        * block_size
    )
    max_steps = args.max_steps or math.ceil(args.target_train_tokens / tokens_per_step)
    save_steps = args.save_steps or max(1, math.ceil(max_steps / args.analysis_checkpoints))
    eval_steps = args.eval_steps or save_steps
    milestones = set(range(0, max_steps + 1, save_steps))
    milestones.add(max_steps)

    config = load_config(args.config)
    config["use_cache"] = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_model(args.variant, **config)
    validate_tokenizer(data_meta, args.tokenizer, tokenizer, model.config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    eval_blocks = max(1, math.ceil(args.trainer_eval_tokens / block_size))
    eval_dataset = data["validation"].select(
        range(min(eval_blocks, len(data["validation"])))
    )

    use_cuda = torch.cuda.is_available()
    precision = mixed_precision_flags(args.bf16, use_cuda)
    tf32 = args.tf32 and use_cuda
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        warmup_steps=min(args.warmup_steps, max_steps),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        tf32=tf32,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=args.dataloader_num_workers,
        disable_tqdm=False,
        prediction_loss_only=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if training_args.process_index == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        manifest = {
            "variant": args.variant,
            "seed": args.seed,
            "parameter_count": parameter_count,
            "data_dir": str(Path(args.data_dir).resolve()),
            "data_meta": data_meta,
            "world_size": world_size,
            "block_size": block_size,
            "tokens_per_step": tokens_per_step,
            "target_train_tokens": args.target_train_tokens,
            "planned_processed_tokens": max_steps * tokens_per_step,
            "max_steps": max_steps,
            "save_steps": save_steps,
            "analysis_milestones": sorted(milestones),
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "packages": package_versions(),
            "cuda_available": use_cuda,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
        with (output_dir / "run_manifest.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)

    snapshot_callback = AnalysisSnapshotCallback(args.output_dir, milestones)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[snapshot_callback],
    )
    print(
        f"[train] variant={args.variant} params={parameter_count/1e6:.2f}M "
        f"steps={max_steps:,} tokens/step={tokens_per_step:,}"
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
        (output_dir / "variant.txt").write_text(args.variant)
        print(f"[train] saved final model -> {output_dir}")

        if args.analyze_after_train:
            from analyze_checkpoints import analyze_run

            analyze_run(
                run_dir=output_dir,
                data_dir=Path(args.data_dir),
                variant=args.variant,
                context_lengths=[
                    int(value) for value in args.analysis_context_lengths.split(",")
                ],
                ppl_tokens=args.analysis_ppl_tokens,
                sink_length=args.analysis_sink_length,
                sink_samples=args.analysis_sink_samples,
                device="cuda" if use_cuda else "cpu",
            )


if __name__ == "__main__":
    main()
