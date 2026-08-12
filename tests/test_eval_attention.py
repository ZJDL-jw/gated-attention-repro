import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "our"))

from eval_attention import compute_text_ppl


class EvalAttentionTest(unittest.TestCase):
    def test_text_ppl_overlaps_chunks_at_the_boundary(self):
        class Tokenizer:
            def __call__(self, _text, return_tensors):
                return SimpleNamespace(input_ids=torch.tensor([[0, 1, 2, 3, 4]]))

        class RecordingModel:
            def __init__(self):
                self.calls = []

            def __call__(self, input_ids, **_kwargs):
                self.calls.append(input_ids[0].tolist())
                return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 5))

        model = RecordingModel()
        compute_text_ppl(model, Tokenizer(), "ignored", "cpu", max_len=3)
        self.assertEqual(model.calls, [[0, 1, 2], [2, 3, 4]])

    def test_official_curve_loader_disables_pickle_execution(self):
        from plot_official_figures import load_curve

        path = Path("curve.pt")
        with mock.patch("plot_official_figures.torch.load", return_value={}) as load:
            load_curve(path)
        load.assert_called_once_with(path, map_location="cpu", weights_only=True)


if __name__ == "__main__":
    unittest.main()
