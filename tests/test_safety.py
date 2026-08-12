import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "our"))


class SafetyTest(unittest.TestCase):
    def test_data_output_rejects_project_root_and_symlink(self):
        from prep_data import safe_output_path

        with self.assertRaises(ValueError):
            safe_output_path(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_output_path(link)

    def test_dataset_publish_does_not_expose_partial_output(self):
        from prep_data import commit_dataset

        class BrokenDataset:
            def save_to_disk(self, path):
                staged = Path(path)
                staged.mkdir()
                (staged / "partial").write_text("incomplete")
                raise OSError("disk full")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "published"
            output.mkdir()
            (output / "old-data").write_text("still valid")
            with self.assertRaises(OSError):
                commit_dataset(BrokenDataset(), root / "staged", output, {})
            self.assertEqual((output / "old-data").read_text(), "still valid")

    def test_dataset_publish_moves_complete_staging_atomically(self):
        from prep_data import commit_dataset

        class CompleteDataset:
            def save_to_disk(self, path):
                staged = Path(path)
                staged.mkdir()
                (staged / "dataset_dict.json").write_text("{}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "published"
            staged = root / "staged"
            output.mkdir()
            (output / "old-data").write_text("replace me")
            commit_dataset(CompleteDataset(), staged, output, {"complete": True})
            self.assertFalse(staged.exists())
            self.assertFalse((output / "old-data").exists())
            self.assertTrue((output / "dataset_dict.json").exists())
            self.assertIn('"complete": true', (output / "meta.json").read_text())

    def test_dataset_publish_rolls_back_if_final_rename_fails(self):
        import os
        import prep_data
        from prep_data import commit_dataset

        class CompleteDataset:
            def save_to_disk(self, path):
                staged = Path(path)
                staged.mkdir()
                (staged / "new-data").write_text("new")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "published"
            output.mkdir()
            (output / "old-data").write_text("old")
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("rename failed")
                return real_replace(source, destination)

            with mock.patch.object(prep_data.os, "replace", fail_second_replace):
                with self.assertRaises(OSError):
                    commit_dataset(CompleteDataset(), root / "staged", output, {})
            self.assertEqual((output / "old-data").read_text(), "old")

    def test_training_rejects_nonempty_output_without_resume(self):
        from train import validate_output_dir

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "old-snapshot").write_text("stale")
            with self.assertRaises(FileExistsError):
                validate_output_dir(output)
            self.assertEqual(
                validate_output_dir(output, resume_from_checkpoint="checkpoint-1"),
                output,
            )

    def test_tokenizer_metadata_mismatch_is_rejected(self):
        from train import validate_tokenizer

        class Tokenizer:
            eos_token_id = 2

            def __len__(self):
                return 5

        tokenizer = Tokenizer()
        with self.assertRaises(ValueError):
            validate_tokenizer(
                {"tokenizer": "prepared", "eos_token_id": 2},
                "different",
                tokenizer,
                SimpleNamespace(vocab_size=8),
            )

    def test_tokenizer_eos_and_vocab_mismatches_are_rejected(self):
        from train import validate_tokenizer

        class Tokenizer:
            eos_token_id = 3

            def __len__(self):
                return 9

        with self.assertRaisesRegex(ValueError, "EOS id"):
            validate_tokenizer(
                {"tokenizer": "same", "eos_token_id": 2},
                "same",
                Tokenizer(),
                SimpleNamespace(vocab_size=10),
            )
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            validate_tokenizer(
                {"tokenizer": "same", "eos_token_id": 3},
                "same",
                Tokenizer(),
                SimpleNamespace(vocab_size=8),
            )

    def test_cuda_precision_falls_back_when_bf16_is_unsupported(self):
        import torch
        from eval_attention import evaluation_dtype
        from train import mixed_precision_flags

        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(evaluation_dtype("cuda"), torch.float16)
            self.assertEqual(
                mixed_precision_flags(True, use_cuda=True),
                {"bf16": False, "fp16": True},
            )


if __name__ == "__main__":
    unittest.main()
