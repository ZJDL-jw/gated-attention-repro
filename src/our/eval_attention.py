"""Phase A evaluation: load a gated-attention model, measure attention sink,
plot attention maps, and compute PPL.

Run on the 4xL20 (CUDA) machine:
    python src/our/eval_attention.py \
        --model_path models_1b/1B_baseline \
        --variant baseline \
        --output_dir results/phaseA \
        --prompt "Sparse gating mechanism mitigates attention sink." \
        --ppl_text data/ppl_sample.txt

Outputs (in --output_dir):
    attention_maps_{variant}.png   : 4-layer attention heatmaps
    results_{variant}.json         : {mean_first_token_rate, per_layer, ppl, ...}
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from model_builder import Qwen3ForCausalLM, Qwen3Config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--variant", required=True, help="baseline | gate_headwise | gate_elementwise")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt", default="Sparse gating mechanism mitigates attention sink.")
    p.add_argument("--ppl_text", default=None, help="path to a plaintext file for PPL")
    p.add_argument("--layers", default="0,6,20,27", help="comma-separated layer indices to visualize")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def load_model(model_path, device):
    cfg = Qwen3Config.from_pretrained(model_path)
    # eager attention is required to extract per-head attention weights
    # (SDPA does not support output_attentions=True).
    model = Qwen3ForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.to(device).eval()
    return model


@torch.no_grad()
def extract(model, input_ids, device):
    out = model(input_ids=input_ids.to(device), output_attentions=True)
    return out.logits, out.attentions  # attentions: tuple[layers] -> (B, H, L, L)


def first_token_rate(attentions):
    """Mean proportion of attention mass placed on the FIRST token.

    For each layer, average the first-key weight over batch, heads and query
    positions. Returns a list (per layer) of scalars in [0, 1].
    """
    rates = []
    for a in attentions:  # (B, H, L, L)
        w = a[:, :, :, 0].float().mean(dim=(0, 1))  # (L,)
        rates.append(float(w.mean()))
    return rates


def plot_maps(attentions, tokens, layers, save_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, li in enumerate(layers):
        a = attentions[li].float().mean(dim=1)[0].cpu().numpy()  # (L, L)
        ax = axes[idx]
        im = ax.imshow(a, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{title} | Layer {li + 1}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


@torch.no_grad()
def compute_ppl(model, tokenizer, text, device, max_len=1024):
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(enc) < 2:
        return None
    loss_f = torch.nn.CrossEntropyLoss(reduction="sum")
    total, count = 0.0, 0
    for start in range(0, len(enc) - 1, max_len):
        chunk = enc[start : start + max_len + 1]
        if len(chunk) < 2:
            break
        inp = chunk[:-1].unsqueeze(0).to(device)
        lab = chunk[1:].unsqueeze(0).to(device)
        logits = model(input_ids=inp).logits
        loss = loss_f(logits.float().view(-1, logits.size(-1)), lab.view(-1))
        total += loss.item()
        count += lab.numel()
    return math.exp(total / count)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    model = load_model(args.model_path, device)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path)
    except Exception:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    inputs = tok(args.prompt, return_tensors="pt")
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

    logits, attentions = extract(model, inputs["input_ids"], device)
    rates = first_token_rate(attentions)
    mean_rate = sum(rates) / len(rates)

    n_layers = len(attentions)
    req = [int(x) for x in args.layers.split(",")]
    plot_layers = [min(i, n_layers - 1) for i in req]
    plot_maps(attentions, tokens, plot_layers,
              os.path.join(args.output_dir, f"attention_maps_{args.variant}.png"),
              title=args.variant)

    result = {
        "variant": args.variant,
        "n_layers": n_layers,
        "per_layer_first_token_rate": rates,
        "mean_first_token_rate": mean_rate,
        "ppl": None,
    }

    if args.ppl_text and os.path.exists(args.ppl_text):
        with open(args.ppl_text) as f:
            text = f.read()
        result["ppl"] = compute_ppl(model, tok, text, device)

    with open(os.path.join(args.output_dir, f"results_{args.variant}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"=== {args.variant} ===")
    print(f"  mean first-token attention rate : {mean_rate:.4f}")
    print(f"  per-layer                      : {[round(r, 4) for r in rates]}")
    print(f"  PPL ({args.ppl_text or 'n/a'})  : {result['ppl']}")
    print(f"  maps -> {os.path.join(args.output_dir, f'attention_maps_{args.variant}.png')}")


if __name__ == "__main__":
    main()
