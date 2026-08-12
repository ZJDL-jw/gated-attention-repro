"""Evaluate official or self-trained checkpoints with one shared protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from metrics import (
    evaluate_attention_and_probes,
    evaluate_context_lengths,
    read_data_meta,
)
from model_builder import Qwen3Config, Qwen3ForCausalLM


def evaluation_dtype(device):
    """Choose a CUDA dtype supported by the selected hardware."""
    device = torch.device(device)
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--prompt", default="Sparse gating mechanism mitigates attention sink."
    )
    parser.add_argument("--ppl_text", default=None, help="legacy smoke-test text")
    parser.add_argument("--eval_data_dir", default=None)
    parser.add_argument("--ppl_context_length", type=int, default=1024)
    parser.add_argument("--ppl_max_tokens", type=int, default=262_144)
    parser.add_argument("--sink_context_length", type=int, default=512)
    parser.add_argument("--sink_samples", type=int, default=16)
    parser.add_argument("--exclude_query_prefix", type=int, default=4)
    parser.add_argument("--layers", default="0,6,20,27")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


@torch.no_grad()
def load_model(model_path, device):
    device = torch.device(device)
    config = Qwen3Config.from_pretrained(model_path)
    dtype = evaluation_dtype(device)
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    return model.to(device).eval()


@torch.no_grad()
def extract(model, input_ids, device):
    output = model(
        input_ids=input_ids.to(device), output_attentions=True, use_cache=False
    )
    return output.logits, output.attentions


def first_token_rate(attentions, exclude_query_prefix=0):
    """Return per-layer mean attention allocated to the first key token."""
    rates = []
    for attention in attentions:
        first_key = attention[:, :, exclude_query_prefix:, 0].float()
        rates.append(float(first_key.mean()))
    return rates


def plot_maps(attentions, tokens, layers, save_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for index, layer_idx in enumerate(layers[:4]):
        attention = attentions[layer_idx].float().mean(dim=1)[0].cpu().numpy()
        axis = axes[index]
        image = axis.imshow(attention, cmap="viridis")
        fig.colorbar(image, ax=axis)
        axis.set_title(f"{title} | Layer {layer_idx + 1}")
        axis.set_xticks(range(len(tokens)))
        axis.set_yticks(range(len(tokens)))
        axis.set_xticklabels(tokens, rotation=90)
        axis.set_yticklabels(tokens)
        axis.tick_params(length=0)
    for axis in axes[len(layers[:4]) :]:
        axis.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


@torch.no_grad()
def compute_text_ppl(model, tokenizer, text, device, max_len=1024):
    """Small backwards-compatible smoke metric; formal runs use a dataset."""
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total_nll = 0.0
    target_tokens = 0
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    for start in range(0, len(token_ids) - 1, max_len - 1):
        chunk = token_ids[start : start + max_len]
        if len(chunk) < 2:
            continue
        input_ids = chunk.unsqueeze(0).to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1]
        labels = input_ids[:, 1:]
        total_nll += float(
            F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
        )
        target_tokens += labels.numel()
    return math.exp(total_nll / target_tokens) if target_tokens else None


def _validation_split(data_dir):
    dataset = load_from_disk(data_dir)
    if isinstance(dataset, DatasetDict):
        if "validation" not in dataset:
            raise KeyError(f"{data_dir} has no validation split")
        return dataset["validation"]
    return dataset


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, args.device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    prompt_inputs = tokenizer(args.prompt, return_tensors="pt")
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_inputs["input_ids"][0])
    _, prompt_attentions = extract(model, prompt_inputs["input_ids"], args.device)
    num_layers = len(prompt_attentions)
    requested_layers = [int(value) for value in args.layers.split(",")]
    plot_layers = [min(value, num_layers - 1) for value in requested_layers]
    map_path = output_dir / f"attention_maps_{args.variant}.png"
    plot_maps(
        prompt_attentions,
        prompt_tokens,
        plot_layers,
        map_path,
        title=args.variant,
    )

    prompt_paper_rates = first_token_rate(prompt_attentions)
    result = {
        "variant": args.variant,
        "model_path": str(Path(args.model_path).resolve()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "n_layers": num_layers,
        "prompt": {
            "text": args.prompt,
            "paper_style_per_layer": prompt_paper_rates,
            "paper_style_mean": sum(prompt_paper_rates) / len(prompt_paper_rates),
        },
        "dataset": None,
        "legacy_text_ppl": None,
    }

    if args.eval_data_dir:
        validation = _validation_split(args.eval_data_dir)
        ppl = evaluate_context_lengths(
            model,
            validation,
            [args.ppl_context_length],
            args.device,
            max_tokens_per_length=args.ppl_max_tokens,
            batch_size=1,
        )[0]
        probe = evaluate_attention_and_probes(
            model,
            validation,
            context_length=args.sink_context_length,
            device=args.device,
            max_samples=args.sink_samples,
            exclude_query_prefix=args.exclude_query_prefix,
        )
        result["dataset"] = {
            "data_dir": str(Path(args.eval_data_dir).resolve()),
            "meta": read_data_meta(args.eval_data_dir),
            "ppl": ppl,
            "probe": probe,
        }
    elif args.ppl_text and Path(args.ppl_text).exists():
        result["legacy_text_ppl"] = compute_text_ppl(
            model,
            tokenizer,
            Path(args.ppl_text).read_text(),
            args.device,
        )

    result_path = output_dir / f"results_{args.variant}.json"
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(f"=== {args.variant} ===")
    if result["dataset"]:
        probe = result["dataset"]["probe"]
        print(f"  validation PPL              : {result['dataset']['ppl']['ppl']:.4f}")
        print(f"  paper-style sink            : {probe['paper_style_mean']:.4f}")
        print(f"  prefix-excluded sink        : {probe['prefix_excluded_mean']:.4f}")
    else:
        print(f"  prompt sink                 : {result['prompt']['paper_style_mean']:.4f}")
        print(f"  legacy text PPL             : {result['legacy_text_ppl']}")
    print(f"  results -> {result_path}")
    print(f"  maps    -> {map_path}")


if __name__ == "__main__":
    main()
