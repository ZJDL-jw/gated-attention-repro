import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "our"))


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
class ModelBuilderTest(unittest.TestCase):
    def test_variant_flags_and_projection_widths(self):
        from model_builder import build_model

        config = dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        baseline = build_model("baseline", **config)
        headwise = build_model("headwise", **config)
        elementwise = build_model("elementwise", **config)
        self.assertEqual(baseline.model.layers[0].self_attn.q_proj.out_features, 32)
        self.assertEqual(headwise.model.layers[0].self_attn.q_proj.out_features, 36)
        self.assertEqual(elementwise.model.layers[0].self_attn.q_proj.out_features, 64)

    def test_config_maps_attention_bias_to_upstream_qkv_bias(self):
        from model_builder import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=True,
        )
        model = Qwen3ForCausalLM(config)
        attention = model.model.layers[0].self_attn
        self.assertTrue(config.qkv_bias)
        self.assertIsNotNone(attention.q_proj.bias)

        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary)
            restored = Qwen3ForCausalLM.from_pretrained(temporary)
            self.assertTrue(restored.config.qkv_bias)

    def test_flash_attention_weights_use_safe_eager_fallback(self):
        import torch
        from official.modeling_qwen3 import Qwen3RotaryEmbedding
        from model_builder import Qwen3Config, SafeQwen3FlashAttention2

        config = Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        attention = SafeQwen3FlashAttention2(config, layer_idx=0)
        hidden = torch.randn(1, 3, 8)
        positions = torch.arange(3).unsqueeze(0)
        embeddings = Qwen3RotaryEmbedding(config=config)(hidden, positions)
        output, weights, _ = attention(
            hidden, None, None, None, True, False, None, embeddings
        )
        self.assertEqual(tuple(output.shape), (1, 3, 8))
        self.assertEqual(tuple(weights.shape), (1, 2, 3, 3))


if __name__ == "__main__":
    unittest.main()
