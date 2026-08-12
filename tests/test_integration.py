import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "our"))


@unittest.skipUnless(
    importlib.util.find_spec("transformers") and importlib.util.find_spec("datasets"),
    "training dependencies not installed",
)
class TrainingIntegrationTest(unittest.TestCase):
    def test_streaming_arrow_builder(self):
        from datasets import Dataset, Features, Sequence, Value
        import prep_data

        class FakeTokenizer:
            eos_token_id = 2

            def __call__(self, text, **_kwargs):
                return {"input_ids": [4] * int(text)}

        source = Dataset.from_dict({"text": ["3", "4", "5"]})
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(prep_data, "_load_source", return_value=source), mock.patch.object(
                prep_data.AutoTokenizer,
                "from_pretrained",
                return_value=FakeTokenizer(),
            ):
                packed = Dataset.from_generator(
                    prep_data.iter_packed_blocks,
                    gen_kwargs={
                        "dataset_name": "fake",
                        "dataset_config": None,
                        "dataset_split": "train",
                        "tokenizer_name": "fake",
                        "text_field": "text",
                        "block_size": 4,
                        "total_blocks": 2,
                        "streaming": False,
                        "seed": 20,
                        "shuffle_buffer": 0,
                    },
                    features=Features(
                        {"input_ids": Sequence(Value("int32"), length=4)}
                    ),
                    cache_dir=temporary,
                )
                self.assertEqual(len(packed), 2)
                self.assertEqual(packed[0]["input_ids"], [4, 4, 4, 2])

    def test_one_step_train_snapshot_and_analysis(self):
        from datasets import Dataset, DatasetDict
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_backend = Tokenizer(
                models.WordLevel(
                    {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "x": 4},
                    unk_token="<unk>",
                )
            )
            tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer_backend,
                pad_token="<pad>",
                bos_token="<bos>",
                eos_token="<eos>",
                unk_token="<unk>",
            )
            tokenizer_dir = root / "tokenizer"
            tokenizer.save_pretrained(tokenizer_dir)

            rows = [[4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4]]
            data = DatasetDict(
                train=Dataset.from_dict({"input_ids": rows * 4}),
                validation=Dataset.from_dict({"input_ids": rows * 2}),
            )
            data_dir = root / "data"
            data.save_to_disk(str(data_dir))
            (data_dir / "meta.json").write_text(
                json.dumps({"block_size": 16, "train_tokens": 64})
            )

            config = {
                "vocab_size": 5,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 16,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            output_dir = root / "run"

            command = [
                "train.py",
                "--variant",
                "elementwise",
                "--config",
                str(config_path),
                "--data_dir",
                str(data_dir),
                "--tokenizer",
                str(tokenizer_dir),
                "--output_dir",
                str(output_dir),
                "--max_steps",
                "1",
                "--per_device_train_batch_size",
                "1",
                "--per_device_eval_batch_size",
                "1",
                "--trainer_eval_tokens",
                "32",
                "--analysis_checkpoints",
                "1",
                "--analysis_context_lengths",
                "8,16",
                "--analysis_ppl_tokens",
                "32",
                "--analysis_sink_length",
                "8",
                "--analysis_sink_samples",
                "1",
                "--dataloader_num_workers",
                "0",
                "--no-bf16",
                "--no-tf32",
            ]
            from train import main as train_main

            with mock.patch.object(sys, "argv", command):
                train_main()

            dynamics = output_dir / "analysis" / "dynamics_elementwise.jsonl"
            self.assertTrue((output_dir / "run_manifest.json").exists())
            self.assertTrue((output_dir / "trainer_state.json").exists())
            self.assertTrue((output_dir / "analysis_snapshots" / "step-00000000").exists())
            self.assertTrue((output_dir / "analysis_snapshots" / "step-00000001").exists())
            self.assertTrue(dynamics.exists())
            rows = [json.loads(line) for line in dynamics.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 1])
            self.assertEqual(set(rows[-1]["ppl_by_context"]), {"8", "16"})

            from compare_runs import main as compare_main

            results_dir = root / "results"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "compare_runs.py",
                    "--out_root",
                    str(root),
                    "--out",
                    str(results_dir),
                ],
            ):
                compare_main()
            self.assertTrue((results_dir / "summary.csv").exists())
            self.assertTrue((results_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
