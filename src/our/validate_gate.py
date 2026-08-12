"""Fast CPU sanity check for all three attention variants."""
from __future__ import annotations

import torch

from model_builder import VARIANTS, build_model


MICRO_CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=64,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    use_cache=False,
)


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


def force_gate_hook(attention, value):
    def hook(_module, _inputs, output):
        output = output.clone()
        grouped = output.view(
            output.size(0), output.size(1), attention.num_key_value_heads, -1
        )
        query_width = attention.head_dim * attention.num_key_value_groups
        grouped[..., query_width:] = value
        return grouped.view_as(output)

    return hook


def main():
    torch.manual_seed(0)
    models = {variant: build_model(variant, **MICRO_CONFIG) for variant in VARIANTS}
    counts = {variant: count_params(model) for variant, model in models.items()}
    assert counts["headwise"] > counts["baseline"]
    assert counts["elementwise"] > counts["headwise"]

    print("=== parameter counts ===")
    for variant in VARIANTS:
        print(f"  {variant:12s} {counts[variant]:,}")

    assert models["baseline"].model.layers[0].self_attn.q_proj.out_features == 64
    assert models["headwise"].model.layers[0].self_attn.q_proj.out_features == 68
    assert models["elementwise"].model.layers[0].self_attn.q_proj.out_features == 128

    input_ids = torch.randint(3, MICRO_CONFIG["vocab_size"], (2, 16))
    for variant, model in models.items():
        output = model(input_ids=input_ids, labels=input_ids, output_attentions=True)
        assert output.logits.shape == (2, 16, MICRO_CONFIG["vocab_size"])
        assert len(output.attentions) == MICRO_CONFIG["num_hidden_layers"]
        assert torch.isfinite(output.loss)
        output.loss.backward()
        print(f"  {variant:12s} loss={float(output.loss):.4f}")

    # Force only the gate logits while keeping every model weight fixed.
    gated = models["elementwise"].eval()
    hooks = []
    with torch.no_grad():
        for layer in gated.model.layers:
            hooks.append(layer.self_attn.q_proj.register_forward_hook(
                force_gate_hook(layer.self_attn, -20.0)
            ))
        closed = gated(input_ids=input_ids, use_cache=False).logits
        for hook in hooks:
            hook.remove()
        hooks = []
        for layer in gated.model.layers:
            hooks.append(layer.self_attn.q_proj.register_forward_hook(
                force_gate_hook(layer.self_attn, 20.0)
            ))
        opened = gated(input_ids=input_ids, use_cache=False).logits
        for hook in hooks:
            hook.remove()
    assert not torch.allclose(closed, opened, atol=1e-5)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
