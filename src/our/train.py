"""Phase B: train a small gated-attention model from scratch and track the
attention-sink metric during training.

This is the heart of the self-reproduction. We train THREE variants
(baseline / headwise / elementwise) from the SAME config and SAME data, then
compare (a) downstream loss, (b) training stability (loss spikes), and
(c) the attention-sink rate measured on a fixed probe prompt every N steps.

Run on the 4xL20 with accelerate (4 processes == 4 GPUs):
    accelerate launch --num_processes 4 src/our/train.py \
        --variant headwise \
        --config configs/qwen3_tiny.json \
        --data_dir data/tinystories_50M \
        --output_dir outputs/headwise

Or a single-GPU smoke test:
    python src/our/train.py --variant baseline \
        --config configs/qwen3_tiny.json \
        --data_dir data/tinystories_50M --max_steps 50 \
        --output_dir /tmp/smoke_baseline

We reuse the attention extractors from eval_attention so train/eval share
exactly the same sink definition.
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from model_builder import build_model, VARIANTS
from eval_attention import first_token_rate, extract, load_model, plot_maps


# --- gate flags must NEVER be overridden by the config file; the variant wins.
GATE_FLAGS = ("headwise_attn_output_gate", "elementwise_attn_output_gate")


def load_config(path, variant):
    with open(path) as f:
        cfg = json.load(f)
    for k in GATE_FLAGS:
        cfg.pop(k, None)  # variant decides these
    return cfg


class SinkTracker(TrainerCallback):
    """Every `every_steps`, run a forward pass on a probe prompt with
    output_attentions=True and record the mean first-token attention rate
    (the attention-sink proxy). Produces sink_curve_{variant}.jsonl."""

    def __init__(self, tokenizer, device, prompt, every_steps, out_jsonl):
        self.tokenizer = tokenizer
        self.device = device
        self.prompt = prompt
        self.every = every_steps
        self.out = out_jsonl
        self.enc = tokenizer(prompt, return_tensors="pt")["input_ids"]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if state.global_step % self.every != 0:
            return control
        dev = next(model.parameters()).device
        ids = self.enc.to(dev)
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True)
        rates = first_token_rate(out.attentions)
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        row = dict(
            step=state.global_step,
            loss=last_loss,
            mean_first_token_rate=sum(rates) / len(rates),
            per_layer=rates,
        )
        with open(self.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        return control


def post_eval(variant, output_dir, prompt, layers):
    """After training, load the saved checkpoint and produce attention maps +
    a results json, identical in format to Phase A (so they compare directly)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(output_dir, device)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    inputs = tok(prompt, return_tensors="pt")
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
    logits, atts = extract(model, inputs["input_ids"], device)
    rates = first_token_rate(atts)
    n_layers = len(atts)
    req = [int(x) for x in layers.split(",")]
    plot_layers = [min(i, n_layers - 1) for i in req]

    res = dict(
        variant=f"{variant}_trained",
        n_layers=n_layers,
        per_layer_first_token_rate=rates,
        mean_first_token_rate=sum(rates) / len(rates),
    )
    with open(os.path.join(output_dir, f"results_{variant}_trained.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot_maps(atts, tokens, plot_layers,
              os.path.join(output_dir, f"attention_maps_{variant}_trained.png"),
              title=f"{variant}_trained")
    print(f"[post_eval:{variant}] mean first-token rate "
          f"{res['mean_first_token_rate']:.4f} -> results_{variant}_trained.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--config", required=True, help="path to qwen3_tiny.json")
    ap.add_argument("--data_dir", required=True, help="output of prep_data.py")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--num_train_epochs", type=float, default=None)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--sink_eval_steps", type=int, default=100,
                    help="how often to probe the attention-sink rate")
    ap.add_argument("--sink_prompt",
                    default="The gating mechanism lets the model decide which "
                            "tokens to attend to, reducing attention sink.")
    ap.add_argument("--layers", default="0,3,7,11",
                    help="layers to visualize in post-eval (tiny model = 12)")
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.variant)
    model = build_model(args.variant, **cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] variant={args.variant}  params={n_params/1e6:.2f}M")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = load_from_disk(args.data_dir)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # training-args: bf16 + multi-GPU via accelerate (num_processes=4).
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        tf32=True,
        report_to="none",
        seed=args.seed,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        disable_tqdm=False,
    )

    sink_jsonl = os.path.join(args.output_dir, f"sink_curve_{args.variant}.jsonl")
    tracker = SinkTracker(tok, "cpu", args.sink_prompt, args.sink_eval_steps, sink_jsonl)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[tracker],
    )
    print(f"[train] starting training on "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'} "
          f"(max_steps={args.max_steps}) ...")
    trainer.train()

    trainer.save_model(args.output_dir)
    # save the *variant config* so evaluation knows which gate is active
    with open(os.path.join(args.output_dir, "variant.txt"), "w") as f:
        f.write(args.variant)
    print(f"[train] saved checkpoint -> {args.output_dir}")

    if trainer.is_world_process_zero:
        post_eval(args.variant, args.output_dir, args.sink_prompt, args.layers)
        print(f"[train] sink curve -> {sink_jsonl}")


if __name__ == "__main__":
    main()
