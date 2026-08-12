import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "our"))

from metrics import (
    _new_gate_stats,
    _new_moment_stats,
    _update_gate_stats,
    _update_moment_stats,
    compute_dataset_ppl,
    extract_gate_values,
    iter_context_blocks,
    iter_scoring_blocks,
    summarize_attentions,
)


class MetricsTest(unittest.TestCase):
    def test_reblocking_uses_identical_token_stream(self):
        dataset = [
            {"input_ids": list(range(8))},
            {"input_ids": list(range(8, 16))},
        ]
        blocks_4 = list(iter_context_blocks(dataset, 4))
        blocks_8 = list(iter_context_blocks(dataset, 8))
        self.assertEqual([token for row in blocks_4 for token in row], list(range(16)))
        self.assertEqual([token for row in blocks_8 for token in row], list(range(16)))

    def test_scoring_blocks_cover_every_transition_once(self):
        dataset = [{"input_ids": list(range(11))}]
        blocks = list(iter_scoring_blocks(dataset, context_length=4))
        self.assertEqual(blocks, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9, 10]])
        self.assertEqual(
            [token for block in blocks for token in block[1:]],
            list(range(1, 11)),
        )
        limited = list(
            iter_scoring_blocks(dataset, context_length=4, max_target_tokens=5)
        )
        self.assertEqual(
            [token for block in limited for token in block[1:]],
            list(range(1, 6)),
        )

    def test_ppl_context_lengths_score_the_same_targets(self):
        class UniformModel(torch.nn.Module):
            def forward(self, input_ids, **_kwargs):
                logits = torch.zeros(*input_ids.shape, 16)
                return SimpleNamespace(logits=logits)

        dataset = [{"input_ids": list(range(11))}]
        model = UniformModel()
        short = compute_dataset_ppl(model, dataset, 4, "cpu")
        long = compute_dataset_ppl(model, dataset, 7, "cpu")
        self.assertEqual(short["target_tokens"], 10)
        self.assertEqual(long["target_tokens"], 10)
        self.assertAlmostEqual(short["ppl"], long["ppl"])

    def test_probe_accumulators_stay_as_tensors_until_finalize(self):
        moments = _new_moment_stats()
        gates = _new_gate_stats()
        _update_moment_stats(moments, torch.tensor([-2.0, 1.0]))
        _update_gate_stats(gates, torch.tensor([0.05, 0.75]))
        self.assertIsInstance(moments["sum"], torch.Tensor)
        self.assertIsInstance(moments["max_abs"], torch.Tensor)
        self.assertIsInstance(gates["lt_0_1"], torch.Tensor)
        self.assertIsInstance(gates["min"], torch.Tensor)

    def test_attention_sink_reports_both_definitions(self):
        attention = torch.zeros(1, 1, 4, 4)
        attention[0, 0, :, 0] = torch.tensor([1.0, 0.5, 0.25, 0.125])
        summary = summarize_attentions((attention,), exclude_query_prefix=1)
        self.assertAlmostEqual(summary["paper_style_mean"], 0.46875)
        self.assertAlmostEqual(
            summary["prefix_excluded_mean"], (0.5 + 0.25 + 0.125) / 3
        )

    def test_gate_extraction_matches_grouped_q_proj_layout(self):
        attention = SimpleNamespace(
            num_key_value_heads=2,
            head_dim=2,
            num_key_value_groups=2,
        )
        # Per KV group: four query values followed by two headwise gate logits.
        projection = torch.tensor(
            [[[10, 11, 12, 13, -2, 2, 20, 21, 22, 23, 0, 1]]],
            dtype=torch.float32,
        )
        gates = extract_gate_values(projection, attention)
        self.assertEqual(tuple(gates.shape), (1, 1, 2, 2))
        expected = torch.sigmoid(torch.tensor([[[-2.0, 2.0], [0.0, 1.0]]]))
        self.assertTrue(torch.allclose(gates, expected))
        self.assertTrue(math.isclose(float(gates[0, 0, 1, 0]), 0.5))


if __name__ == "__main__":
    unittest.main()
