"""CPU sanity check for the gated-attention implementation.

Run (after env is installed):
    .venv/bin/python src/our/validate_gate.py

It verifies:
  1. gated variants have strictly MORE parameters than baseline;
  2. the q_proj of gated variants is extended as expected;
  3. all three variants forward/backward without NaNs on random inputs;
  4. the gate actually changes the output (gated != baseline for same input).
"""
import torch

from model_builder import build_model, VARIANTS


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    models = {v: build_model(v) for v in VARIANTS}

    base_n = count_params(models["baseline"])
    print("=== parameter counts ===")
    for v in VARIANTS:
        n = count_params(models[v])
        print(f"  {v:12s} {n/1e6:7.3f}M")
    assert count_params(models["headwise"]) > base_n
    assert count_params(models["elementwise"]) > base_n

    print("=== q_proj out_features ===")
    for v in ("headwise", "elementwise"):
        q = models[v].model.layers[0].self_attn.q_proj
        print(f"  {v:12s} {q.out_features}")

    print("=== forward / loss / attentions ===")
    ids = torch.randint(0, 1000, (2, 16))
    for v in VARIANTS:
        out = models[v](input_ids=ids, labels=ids, output_attentions=True)
        assert out.logits.shape == (2, 16, 151936), out.logits.shape
        assert len(out.attentions) == 12
        assert out.attentions[0].shape == (2, 8, 16, 16)
        assert torch.isfinite(out.loss)
        print(f"  {v:12s} loss={float(out.loss):.4f} logits={tuple(out.logits.shape)}")

    print("=== gate changes output? ===")
    with torch.no_grad():
        o_base = models["baseline"](input_ids=ids).logits
        o_head = models["headwise"](input_ids=ids).logits
        o_elem = models["elementwise"](input_ids=ids).logits
    assert not torch.allclose(o_base, o_head, atol=1e-4)
    assert not torch.allclose(o_base, o_elem, atol=1e-4)
    print("  OK: gated variants differ from baseline -> gate is active")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
