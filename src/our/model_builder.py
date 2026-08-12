"""Build gated-attention Qwen3 variants from the official implementation.

The official repo ships `modeling_qwen3.py` + `configuration_qwen3.py` as a
*package* (they use a relative import `from .configuration_qwen3 import ...`).
We therefore expose them as the `official` package by adding `src/` to sys.path,
then import `official.modeling_qwen3` / `official.configuration_qwen3`.

Three variants are produced by flipping two config flags:
  - baseline    : headwise=False, elementwise=False  (plain Qwen3 attention)
  - headwise    : headwise=True   (per-head scalar sigmoid gate)
  - elementwise : elementwise=True (per-element sigmoid gate)
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from official.configuration_qwen3 import Qwen3Config as _OfficialQwen3Config  # noqa: E402
from official.modeling_qwen3 import (  # noqa: E402
    QWEN3_ATTENTION_CLASSES,
    Qwen3Attention,
    Qwen3FlashAttention2,
    Qwen3ForCausalLM as _OfficialQwen3ForCausalLM,
)


class Qwen3Config(_OfficialQwen3Config):
    """Project-compatible config for the pinned upstream implementation.

    The upstream model reads ``config.qkv_bias`` while its config only declares
    ``attention_bias``. Keep both names synchronized so configs created locally
    and checkpoints containing either spelling work consistently.
    """

    def __init__(self, *args, qkv_bias=None, attention_bias=False, **kwargs):
        if qkv_bias is None:
            qkv_bias = attention_bias
        super().__init__(*args, attention_bias=attention_bias, **kwargs)
        self.qkv_bias = qkv_bias
        self.attention_bias = qkv_bias


class Qwen3ForCausalLM(_OfficialQwen3ForCausalLM):
    """Use the compatible config when loading checkpoints through this project."""

    config_class = Qwen3Config


class SafeQwen3FlashAttention2(Qwen3FlashAttention2):
    """Fall back to eager attention when callers request attention weights."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        arguments = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if output_attentions:
            return Qwen3Attention.forward(
                self,
                output_attentions=True,
                **arguments,
            )
        return super().forward(
            output_attentions=False,
            **arguments,
        )


# Every project model is constructed after importing this module, so replace
# the broken upstream FlashAttention class in the factory mapping once.
QWEN3_ATTENTION_CLASSES["flash_attention_2"] = SafeQwen3FlashAttention2

VARIANTS = ("baseline", "headwise", "elementwise")

# A small, faithful Qwen3-style config. Tokenizer is the official Qwen3
# tokenizer (vocab 151936); we keep hidden_size modest (512) so the total
# parameter count stays ~200M and trains in minutes on a single L20.
DEFAULT_CONFIG = dict(
    vocab_size=151936,
    hidden_size=512,
    intermediate_size=1376,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=4,        # GQA (like Qwen3)
    head_dim=64,
    max_position_embeddings=2048,
    use_qk_norm=True,             # Qwen3 uses qk-norm
    use_sliding_window=False,
    attention_bias=False,
    qkv_bias=False,  # repo modeling reads config.qkv_bias for q/k/v/o projections
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    tie_word_embeddings=False,
    # Token ids are required by modern transformers (the repo's Qwen3Config no
    # longer sets them implicitly). Flow through **kwargs -> super().__init__.
    pad_token_id=0,
    bos_token_id=151643,
    eos_token_id=151645,
    headwise_attn_output_gate=False,
    elementwise_attn_output_gate=False,
)


def make_config(variant: str, **overrides) -> Qwen3Config:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)  # user overrides first...
    # ...then the variant's gate flag ALWAYS wins (never let a config file
    # accidentally disable the very gate we are trying to study).
    if variant == "headwise":
        cfg["headwise_attn_output_gate"] = True
        cfg["elementwise_attn_output_gate"] = False
    elif variant == "elementwise":
        cfg["elementwise_attn_output_gate"] = True
        cfg["headwise_attn_output_gate"] = False
    else:
        cfg["headwise_attn_output_gate"] = False
        cfg["elementwise_attn_output_gate"] = False
    return Qwen3Config(**cfg)


def build_model(variant: str, **overrides) -> Qwen3ForCausalLM:
    cfg = make_config(variant, **overrides)
    return Qwen3ForCausalLM(cfg)


if __name__ == "__main__":
    for v in VARIANTS:
        m = build_model(v)
        n = sum(p.numel() for p in m.parameters())
        print(f"{v:12s} -> {n/1e6:6.2f}M params")
