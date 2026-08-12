"""Shared, dataset-level metrics for both official and self-trained models."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F


def read_data_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def iter_context_blocks(
    dataset: Iterable[dict],
    context_length: int,
    max_tokens: int | None = None,
) -> Iterator[list[int]]:
    """Reblock one token stream without changing the underlying token order."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    buffer: list[int] = []
    emitted_tokens = 0
    for row in dataset:
        buffer.extend(row["input_ids"])
        while len(buffer) >= context_length:
            if max_tokens is not None and emitted_tokens + context_length > max_tokens:
                return
            yield buffer[:context_length]
            del buffer[:context_length]
            emitted_tokens += context_length


def iter_context_batches(
    dataset: Iterable[dict],
    context_length: int,
    batch_size: int = 1,
    max_tokens: int | None = None,
) -> Iterator[torch.Tensor]:
    batch: list[list[int]] = []
    for block in iter_context_blocks(dataset, context_length, max_tokens):
        batch.append(block)
        if len(batch) == batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
    if batch:
        yield torch.tensor(batch, dtype=torch.long)


def iter_scoring_blocks(
    dataset: Iterable[dict],
    context_length: int,
    max_target_tokens: int | None = None,
) -> Iterator[list[int]]:
    """Yield overlapping blocks so every stream transition is scored once."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if max_target_tokens is not None and max_target_tokens < 1:
        return

    buffer: list[int] = []
    scored = 0
    for row in dataset:
        buffer.extend(row["input_ids"])
        while len(buffer) >= context_length:
            remaining = (
                context_length - 1
                if max_target_tokens is None
                else min(context_length - 1, max_target_tokens - scored)
            )
            if remaining <= 0:
                return
            yield buffer[: remaining + 1]
            del buffer[:remaining]
            scored += remaining
            if remaining < context_length - 1:
                return

    if len(buffer) >= 2:
        remaining = len(buffer) - 1
        if max_target_tokens is not None:
            remaining = min(remaining, max_target_tokens - scored)
        if remaining > 0:
            yield buffer[: remaining + 1]


def iter_scoring_batches(
    dataset: Iterable[dict],
    context_length: int,
    batch_size: int = 1,
    max_target_tokens: int | None = None,
) -> Iterator[torch.Tensor]:
    """Batch equal-length scoring blocks, flushing a shorter final block."""
    batch: list[list[int]] = []
    batch_length = None
    for block in iter_scoring_blocks(dataset, context_length, max_target_tokens):
        if batch and len(block) != batch_length:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
        batch.append(block)
        batch_length = len(block)
        if len(batch) == batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
            batch_length = None
    if batch:
        yield torch.tensor(batch, dtype=torch.long)


@torch.no_grad()
def compute_dataset_ppl(
    model,
    dataset,
    context_length: int,
    device: str | torch.device,
    max_tokens: int | None = None,
    batch_size: int = 1,
) -> dict:
    """Compute token-weighted NLL while scoring each stream transition once."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    target_tokens = 0
    sequences = 0
    try:
        for input_ids in iter_scoring_batches(
            dataset,
            context_length,
            batch_size=batch_size,
            max_target_tokens=max_tokens,
        ):
            input_ids = input_ids.to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits
            labels = input_ids[:, 1:]
            shifted = logits[:, :-1, :]
            nll = F.cross_entropy(
                shifted.float().reshape(-1, shifted.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            target_tokens += labels.numel()
            sequences += input_ids.size(0)
    finally:
        model.train(was_training)

    if target_tokens == 0:
        raise ValueError("Evaluation dataset did not contain a scorable transition")
    mean_nll = total_nll / target_tokens
    return {
        "context_length": context_length,
        "nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "target_tokens": target_tokens,
        "sequences": sequences,
    }


def _new_moment_stats() -> dict:
    return {"count": 0, "sum": None, "sum_sq": None, "max_abs": None}


def _update_moment_stats(stats: dict, value: torch.Tensor) -> None:
    value = value.detach().float()
    absolute = value.abs()
    value_sum = absolute.sum()
    value_sum_sq = value.square().sum()
    value_max = absolute.max()
    stats["count"] += value.numel()
    stats["sum"] = value_sum if stats["sum"] is None else stats["sum"] + value_sum
    stats["sum_sq"] = (
        value_sum_sq
        if stats["sum_sq"] is None
        else stats["sum_sq"] + value_sum_sq
    )
    stats["max_abs"] = (
        value_max
        if stats["max_abs"] is None
        else torch.maximum(stats["max_abs"], value_max)
    )


def _finalize_moment_stats(stats: dict) -> dict:
    count = stats["count"]
    if count == 0:
        return {"mean_abs": None, "rms": None, "max_abs": None, "count": 0}
    total = float(stats["sum"])
    total_sq = float(stats["sum_sq"])
    return {
        "mean_abs": total / count,
        "rms": math.sqrt(total_sq / count),
        "max_abs": float(stats["max_abs"]),
        "count": count,
    }


def _new_gate_stats() -> dict:
    return {
        "count": 0,
        "sum": None,
        "sum_sq": None,
        "lt_0_1": None,
        "lt_0_5": None,
        "min": None,
        "max": None,
    }


def _update_gate_stats(stats: dict, gate: torch.Tensor) -> None:
    gate = gate.detach().float()
    updates = {
        "sum": gate.sum(),
        "sum_sq": gate.square().sum(),
        "lt_0_1": (gate < 0.1).sum(),
        "lt_0_5": (gate < 0.5).sum(),
    }
    stats["count"] += gate.numel()
    for key, update in updates.items():
        stats[key] = update if stats[key] is None else stats[key] + update
    gate_min = gate.min()
    gate_max = gate.max()
    stats["min"] = gate_min if stats["min"] is None else torch.minimum(stats["min"], gate_min)
    stats["max"] = gate_max if stats["max"] is None else torch.maximum(stats["max"], gate_max)


def _finalize_gate_stats(stats: dict) -> dict:
    count = stats["count"]
    if count == 0:
        return {
            "mean": None,
            "std": None,
            "fraction_lt_0_1": None,
            "fraction_lt_0_5": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    total = float(stats["sum"])
    total_sq = float(stats["sum_sq"])
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "fraction_lt_0_1": int(stats["lt_0_1"]) / count,
        "fraction_lt_0_5": int(stats["lt_0_5"]) / count,
        "min": float(stats["min"]),
        "max": float(stats["max"]),
        "count": count,
    }


class ModelProbe(AbstractContextManager):
    """Collect gate and activation statistics without changing official code."""

    ACTIVATION_POINTS = (
        "sdpa_post_gate",
        "attn_output",
        "attn_residual",
        "ffn_output",
        "ffn_residual",
    )

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.gates = defaultdict(_new_gate_stats)
        self.activations = {
            point: defaultdict(_new_moment_stats) for point in self.ACTIVATION_POINTS
        }

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn
            if attention.headwise_attn_output_gate or attention.elementwise_attn_output_gate:
                self.handles.append(
                    attention.q_proj.register_forward_hook(
                        self._gate_hook(layer_idx, attention)
                    )
                )
            self.handles.append(
                attention.o_proj.register_forward_pre_hook(
                    self._activation_pre_hook("sdpa_post_gate", layer_idx)
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._activation_hook("attn_output", layer_idx, tuple_index=0)
                )
            )
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._activation_pre_hook("attn_residual", layer_idx)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._activation_hook("ffn_output", layer_idx)
                )
            )
            self.handles.append(
                layer.register_forward_hook(
                    self._activation_hook("ffn_residual", layer_idx, tuple_index=0)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _gate_hook(self, layer_idx, attention):
        def hook(_module, _inputs, output):
            gate = extract_gate_values(output, attention)
            _update_gate_stats(self.gates[layer_idx], gate)

        return hook

    def _activation_pre_hook(self, point, layer_idx):
        def hook(_module, inputs):
            _update_moment_stats(self.activations[point][layer_idx], inputs[0])

        return hook

    def _activation_hook(self, point, layer_idx, tuple_index=None):
        def hook(_module, _inputs, output):
            value = output[tuple_index] if tuple_index is not None else output
            _update_moment_stats(self.activations[point][layer_idx], value)

        return hook

    def summary(self) -> dict:
        gate_layers = {
            str(layer): _finalize_gate_stats(stats)
            for layer, stats in sorted(self.gates.items())
        }
        all_gates = _new_gate_stats()
        for stats in self.gates.values():
            all_gates["count"] += stats["count"]
            for key in ("sum", "sum_sq", "lt_0_1", "lt_0_5"):
                all_gates[key] = (
                    stats[key]
                    if all_gates[key] is None
                    else all_gates[key] + stats[key]
                )
            all_gates["min"] = (
                stats["min"]
                if all_gates["min"] is None
                else torch.minimum(all_gates["min"], stats["min"])
            )
            all_gates["max"] = (
                stats["max"]
                if all_gates["max"] is None
                else torch.maximum(all_gates["max"], stats["max"])
            )

        activation_summary = {}
        for point, layer_stats in self.activations.items():
            activation_summary[point] = {
                str(layer): _finalize_moment_stats(stats)
                for layer, stats in sorted(layer_stats.items())
            }
        return {
            "gate": {
                "overall": _finalize_gate_stats(all_gates),
                "per_layer": gate_layers,
            },
            "activations": activation_summary,
        }


def extract_gate_values(q_projection: torch.Tensor, attention) -> torch.Tensor:
    """Extract sigmoid gate values from the official extended q_proj layout."""
    batch, length, _ = q_projection.shape
    grouped = q_projection.view(batch, length, attention.num_key_value_heads, -1)
    query_width = attention.head_dim * attention.num_key_value_groups
    return torch.sigmoid(grouped[..., query_width:])


def _new_attention_stats(num_layers: int) -> list[dict]:
    return [
        {"paper_sum": 0.0, "paper_count": 0, "trim_sum": 0.0, "trim_count": 0}
        for _ in range(num_layers)
    ]


def _update_attention_stats(stats, attentions, exclude_query_prefix: int) -> None:
    for layer_idx, attention in enumerate(attentions):
        first_key = attention[:, :, :, 0].detach().float()
        stats[layer_idx]["paper_sum"] += float(first_key.sum())
        stats[layer_idx]["paper_count"] += first_key.numel()
        trimmed = first_key[:, :, exclude_query_prefix:]
        if trimmed.numel():
            stats[layer_idx]["trim_sum"] += float(trimmed.sum())
            stats[layer_idx]["trim_count"] += trimmed.numel()


def _finalize_attention_stats(stats) -> dict:
    paper = [row["paper_sum"] / row["paper_count"] for row in stats]
    trimmed = [
        row["trim_sum"] / row["trim_count"] if row["trim_count"] else None
        for row in stats
    ]
    valid_trimmed = [value for value in trimmed if value is not None]
    return {
        "paper_style_mean": sum(paper) / len(paper),
        "paper_style_per_layer": paper,
        "prefix_excluded_mean": (
            sum(valid_trimmed) / len(valid_trimmed) if valid_trimmed else None
        ),
        "prefix_excluded_per_layer": trimmed,
    }


def summarize_attentions(attentions, exclude_query_prefix: int = 4) -> dict:
    """Summarize one batch of attention tensors using both sink definitions."""
    stats = _new_attention_stats(len(attentions))
    _update_attention_stats(stats, attentions, exclude_query_prefix)
    return _finalize_attention_stats(stats)


@torch.no_grad()
def evaluate_attention_and_probes(
    model,
    dataset,
    context_length: int,
    device: str | torch.device,
    max_samples: int = 16,
    exclude_query_prefix: int = 4,
) -> dict:
    """Measure sink, gate sparsity, and activations on the same token blocks."""
    was_training = model.training
    model.eval()
    attention_stats = None
    samples = 0
    with ModelProbe(model) as probe:
        try:
            for input_ids in iter_context_batches(
                dataset,
                context_length=context_length,
                batch_size=1,
                max_tokens=max_samples * context_length,
            ):
                output = model(
                    input_ids=input_ids.to(device),
                    output_attentions=True,
                    use_cache=False,
                )
                if attention_stats is None:
                    attention_stats = _new_attention_stats(len(output.attentions))
                _update_attention_stats(
                    attention_stats, output.attentions, exclude_query_prefix
                )
                samples += input_ids.size(0)
        finally:
            model.train(was_training)

        if attention_stats is None:
            raise ValueError("No complete blocks were available for attention evaluation")
        result = _finalize_attention_stats(attention_stats)
        result.update(probe.summary())
        result["context_length"] = context_length
        result["samples"] = samples
        result["exclude_query_prefix"] = exclude_query_prefix
        return result


def evaluate_context_lengths(
    model,
    dataset,
    context_lengths: Sequence[int],
    device: str | torch.device,
    max_tokens_per_length: int,
    batch_size: int = 1,
) -> list[dict]:
    return [
        compute_dataset_ppl(
            model,
            dataset,
            context_length=length,
            device=device,
            max_tokens=max_tokens_per_length,
            batch_size=batch_size,
        )
        for length in context_lengths
    ]
